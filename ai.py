# Copyright (c) 2020 PHYTEC Messtechnik GmbH
# SPDX-License-Identifier: Apache-2.0

import os
import time
import cv2

import tflite_runtime.interpreter as tflite
import numpy as np
import json
import concurrent.futures


class Ai:
    def __init__(self, model_path, embeddings_path, modeltype='quant'):
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self.modeltype = modeltype
        self.width = 224
        self.height = 224

    def initialize(self):
        start = time.time()

        self.init_tflite()

        print('Create Embeddigns')
        with open(self.embeddings_path, 'r') as f:
            embeddings_data = json.load(f)

        data = embeddings_data['Embedding']
        self.embeddings = [np.array(data[str(i)]) for i in range(len(data))]

        data = embeddings_data['Name']
        self.names = [np.array(data[str(i)]) for i in range(len(data))]

        data = embeddings_data['File']
        self.files = [np.array(data[str(i)]) for i in range(len(data))]

        self.celeb_embeddings = self.split_data_frame(
                                      self.embeddings,
                                      int(np.ceil(len(self.embeddings)/4)))

        print('Initialization done (duration: {})'.format(time.time() - start))

    def run_inference(self, face, npu=True):
        #Resize face
        print('Resize face')
        if face.shape > (self.width, self.height):
            face = cv2.resize(face, (self.width, self.height),
                              interpolation=cv2.INTER_AREA)
        elif face.shape < (self.width, self.height):
            face = cv2.resize(face, (self.width, self.height),
                              interpolation=cv2.INTER_CUBIC)

        print('Preprocess')
        if self.modeltype == 'quant':
            face = face.astype('float32')
            samples = np.expand_dims(face, axis=0)
            samples = self.preprocess_input(samples).astype('int8')
        else:
            face = face.astype('float32')
            samples = np.expand_dims(face, axis=0)
            samples = self.preprocess_input(samples)

        output_data = self.run_tflite(samples, npu=npu)

        print('Create EUdist')
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            result_1 = executor.submit(self.faceembedding, output_data,
                                       np.array(self.celeb_embeddings[0]))
            result_2 = executor.submit(self.faceembedding, output_data,
                                       np.array(self.celeb_embeddings[1]))
            result_3 = executor.submit(self.faceembedding, output_data,
                                       np.array(self.celeb_embeddings[2]))
            result_4 = executor.submit(self.faceembedding, output_data,
                                       np.array(self.celeb_embeddings[3]))

        EUdist = []
        if result_1.done() & result_2.done() & result_3.done() & result_4.done():
            EUdist.extend(result_1.result())
            EUdist.extend(result_2.result())
            EUdist.extend(result_3.result())
            EUdist.extend(result_4.result())

        idx = np.argpartition(EUdist, 5)
        idx = idx[:5]

        top5 = dict()
        for id in idx:
            top5[id] = [EUdist[id], self.names[id], self.files[id]]

        top5 = {key: value for key, value in sorted(top5.items(), key=lambda item: item[1][0])}

        print('EUdist duration: {}'.format(time.time() - start))

        return top5

    def init_tflite(self):

        os.environ['VIV_VX_CACHE_BINARY_GRAPH_DIR'] = os.getcwd()
        os.environ['VIV_VX_ENABLE_CACHE_GRAPH_BINARY'] = '1'
        ext_delegate= '/usr/lib/libvx_delegate.so'
        ext_delegate= [ tflite.load_delegate(ext_delegate)]

        try:
            self.cpu_interpreter = tflite.Interpreter(self.model_path)
            self.npu_interpreter = tflite.Interpreter(self.model_path, experimental_delegates=ext_delegate)
        except ValueError as e:
            print('Failed to find model file: ' + str(e))
            return

        print('Allocate Tensors')
        self.cpu_interpreter.allocate_tensors()
        self.input_details = self.cpu_interpreter.get_input_details()
        self.output_details = self.cpu_interpreter.get_output_details()

        self.npu_interpreter.allocate_tensors()
        self.input_details = self.npu_interpreter.get_input_details()
        self.output_details = self.npu_interpreter.get_output_details()

    def run_tflite(self, samples, npu):
        print('Invoke TFlite')
        start = time.time()

        if npu:
            interpreter = self.npu_interpreter
        else:
            interpreter = self.cpu_interpreter

        interpreter.set_tensor(self.input_details[0]['index'], samples)
        interpreter.invoke()
        output_data = interpreter.get_tensor(
                        self.output_details[0]['index'])
        print('Interpreter done ({})'.format(time.time() - start))
        return output_data

    def split_data_frame(self, df, chunk_size):
        list_of_df = list()
        number_chunks = len(df) // chunk_size + 1
        for i in range(number_chunks):
            list_of_df.append(df[i*chunk_size:(i+1)*chunk_size])

        return list_of_df

    def preprocess_input(self, x):
        x_temp = np.copy(x)
        x_temp /=255.
        return x_temp

    def faceembedding(self, face, celebdata):
        dist = []
        for i in range(len(celebdata)):
            celebs = np.array(celebdata[i])
            dist.append(np.linalg.norm(face - celebs))

        return dist

