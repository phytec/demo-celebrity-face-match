project(
  'demo-celebrity-face-match',
  version: '0.4',
  default_options: [
    'warning_level=2'
  ]
)

dependency('gtk+-3.0')

python = import('python')
py_installation = python.find_installation('python3')
python_dir = join_paths(get_option('prefix'), py_installation.get_install_dir())
pkg_data_dir = get_option('prefix') / get_option('datadir') / meson.project_name()

conf = configuration_data()
conf.set('PKG_DATA_DIR', pkg_data_dir)

subdir('src')
