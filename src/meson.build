module_dir = python_dir / 'demo_celebrity_face_match'

sources = [
  '__init__.py',
  'ai.py',
  'aidemo.py',
  'camera.py',
  'loadscreen.py',
  configure_file(input: 'config.py.in', output: 'config.py', configuration: conf)
]

install_data(sources, install_dir: module_dir)
install_data(
  'demo-celebrity-face-match',
  install_dir: get_option('bindir'),
  install_mode: 'rwxr-xr-x'
)
