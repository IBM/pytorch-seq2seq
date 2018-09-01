# pytorch-seq2seq Documentation

Welcome to the documentation folder of the pytorch-seq2seq library. We use [sphinx_rtd_theme](https://github.com/rtfd/sphinx_rtd_theme) in our docs. We appreciate even the smallest contribution in making our documentation better and easier to understand.

### Getting Started 

In order to build docs using `sphinx` you need to install a couple of things, make sure you're in the `docs/_dev` folder:

```
pip install -r requirements.txt
```

Once that's out of the way, you can go ahead and build the docs using this command:

```
sphinx-build ./source ./build
```

*Note*: Make sure you've installed the library on your system using the installation guide [here](https://github.com/IBM/pytorch-seq2seq#installation) before building the docs. You'll need to reinstall the package after making changes in the documentation of the code.

Now the docs are ready to be published, so we can move the contents of the `build` dir to the `docs` root dir since github does'nt like to look any further than that. That's it, you've officially made our users' life easier.