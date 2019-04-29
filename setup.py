from setuptools import setup

setup(
      name = "LDApackage",
      version = "1.0",
      author='Xinghong Tang, Jingyi Zhang',
      url='https://github.com/rebeccazjy425/663_LDA_Tang-Zhang.git',
      py_modules = ['LDApackage'],
      install_requires=['jieba','gensim','nltk','nltk.stem','nltk.stem.porter','GibbsSamplingDMM']
      )
