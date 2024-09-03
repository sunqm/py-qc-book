from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    pass

class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension):
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        # Process cmake configuration
        cmd = ['cmake', f'-B{self.build_temp}']
        self.spawn(cmd)

        # Process building
        cmd = ['cmake', '--build', self.build_temp]
        if hasattr(self, "parallel") and self.parallel:
            cmd += [f"-j{self.parallel}"]
        self.spawn(cmd)

setup(
    ext_modules=[
        CMakeExtension('pycopy', sources=[]),
    ],
    cmdclass={"build_ext": CMakeBuild},
    package_data={'pycopy': ['CMakeLists.txt']},
)

