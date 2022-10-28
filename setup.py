from setuptools import setup

APP = ['launch_gui.py']
DATA_FILES = ['models/knn_hog_GT_85.sav', 'models/knn_hog_trio_2_77.sav','models/svm_hog_trio_3_82.sav']
OPTIONS = {
 'packages': ['launch_gui'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    scripts=['launch_gui.py', 'pollen_classification_2.py']
)