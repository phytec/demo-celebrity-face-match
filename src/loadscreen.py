# Copyright (c) 2020 PHYTEC Messtechnik GmbH
# SPDX-License-Identifier: Apache-2.0

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk


class LoadScreen(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title='Load Celebrity Face Match Demo')

        self.set_default_size(640, 480)
        self.set_position(Gtk.WindowPosition.CENTER)
        self.set_border_width(10)

        self.label = Gtk.Label()
        self.progress_bar = Gtk.ProgressBar()
        self.textview = Gtk.TextView()

        self.label.set_markup(
            '<span font="20" font_weight="bold"> Loading Demo ... (can take up to two minutes) </span>'
        )

        self.progress_bar.set_text('ProgressBar')
        self.progress_bar.set_fraction(0.0)

        self.textview.set_editable(True)
        self.textview.set_cursor_visible(True)
        self.textview.set_justification(Gtk.Justification.LEFT)
        self.textview.set_wrap_mode(Gtk.WrapMode.WORD)
        self.textview.set_vexpand(True)
        self.textview.set_monospace(True)
        self.textbuffer = self.textview.get_buffer()

        scrollwindow = Gtk.ScrolledWindow()
        scrollwindow.set_hexpand(True)
        scrollwindow.set_vexpand(True)
        scrollwindow.add(self.textview)

        hbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=30)
        hbox.pack_start(self.label, False, False, 0)
        hbox.pack_start(self.progress_bar, False, False, 0)
        hbox.pack_start(scrollwindow, True, True, 0)

        self.add(hbox)
        self.show_all()

    def append_text(self, text, fraction):
        cur_iter = self.textbuffer.get_end_iter()
        self.textbuffer.insert(cur_iter, text)
        cur_iter = self.textbuffer.get_end_iter()
        self.textbuffer.insert(cur_iter, '\n')
        self.progress_bar.set_fraction(fraction)
