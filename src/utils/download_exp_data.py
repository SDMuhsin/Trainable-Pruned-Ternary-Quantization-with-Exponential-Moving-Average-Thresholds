#!/usr/bin/env python3
"""
    This code downloads some useful data to run some experiments. It avoids
    to relaunch all the computations so some experiments can be done faster.
"""
import os
import requests
from tqdm import tqdm

def download_file(file_url, dir_store, file_name, verbose=True):
    """
        Download a file from the given url and stores it in the given directory
        with the given name
        Arguments:
        ----------
        file_url: str
            Url to the file that we want to download
        dir_store: str
            Path of the directory where the downloaded file should be stored.
        file_name: str
            Name of the file when download it
        verbose: bool
            True if you want to print some information about the requested file
    """
    # Searching if the directory name exists
    if os.path.exists(dir_store+'/'+file_name) and verbose:
        print("The file {} already exists".format(dir_store+'/'+file_name))
    else:
        # Downloading the file
        file = requests.get(file_url)
        open(dir_store+'/'+file_name, 'wb').write(file.content)
        if (verbose):
            print("File downloaded successfully and stored in {}".format(dir_store+'/'+file_name))

def download_ESR():
    """
        Downloads the ESR dataset necessary to run the different
        experiments.
    """
    if (not os.path.exists('../data/EEG_Epileptic_Seizure_Recognition/')):
        print("\n=======> Starting download of Esr dataset")
        # Creating the directories
        os.mkdir('../data/EEG_Epileptic_Seizure_Recognition/')

        # Downloading the different files
        # Data description file
        download_file(
                        file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/data/EEG_Epileptic_Seizure_Recognition/data.hdf5',\
                        dir_store='../data/EEG_Epileptic_Seizure_Recognition/',\
                        file_name='data.hdf5',\
                        verbose=False
                    )
        # Data csv file
        download_file(
                        file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/data/EEG_Epileptic_Seizure_Recognition/Epileptic Seizure Recognition.csv',\
                        dir_store='../data/EEG_Epileptic_Seizure_Recognition/',\
                        file_name='Epileptic Seizure Recognition.csv',\
                        verbose=False
                    )
        print("=======> EEG_Epileptic_Seizure_Recognition dataset downloaded successfully!\n")
    else:
        print("=======> EEG_Epileptic_Seizure_Recognition dataset has alredy been downloaded!\n")



def download_FP_models(model_to_use='mnist2dcnn', dataset='mnist'):
    """
        Downloads the full precision pre-trained model of a given dataset.

        Arguments:
        ----------
        model_to_use: str
            Model that should be downloaded. Three options: mnist2dcnn,
            timefrequency2dcnn, and rawaudiomultichannelcnn.
        dataset: str
            Dataset from which the model was trained. Two options: ESR and MNIST.
    """
    # Creating the folder that wil contain the data  (if it does not exists)
    if (dataset.lower() == 'mnist'):
        if (model_to_use.lower() == 'mnist2dcnn'):
            if not (os.path.exists('../../results/MNIST_2D_CNN_FP/')):
                # Creating useful sub-folders
                os.mkdir('../../results/MNIST_2D_CNN_FP/')
                os.mkdir('../../results/MNIST_2D_CNN_FP/metrics/')
                os.mkdir('../../results/MNIST_2D_CNN_FP/model/')
                os.mkdir('../../results/MNIST_2D_CNN_FP/params_exp/')

                # Downloading files
                # Metrics an models files
                for i in range(10):
                    # Metrics
                    metric_file_to_download = 'results_exp-Exp_MNIST_FullPrecision_rep-{}.pth'.format(i)
                    download_file(
                                    file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/MNIST_2D_CNN_FP/metrics/{}'.format(metric_file_to_download),\
                                    dir_store='../../results/MNIST_2D_CNN_FP/metrics/',\
                                    file_name=metric_file_to_download
                                )
                    # Model
                    model_file_to_download = 'final_model-Exp_MNIST_FullPrecision_rep-{}'.format(i)
                    download_file(
                                    file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/MNIST_2D_CNN_FP/model/{}'.format(model_file_to_download),\
                                    dir_store='../../results/MNIST_2D_CNN_FP/model/',\
                                    file_name=model_file_to_download
                                )
                # Params files
                # Params of the experiment
                download_file(
                                file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/MNIST_2D_CNN_FP/params_exp/params_0.pth',\
                                dir_store='../../results/MNIST_2D_CNN_FP/params_exp/',\
                                file_name='params_0.pth'
                            )
                # Network architecture
                download_file(
                                file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/MNIST_2D_CNN_FP/params_exp/network_architecture.py',\
                                dir_store='../../results/MNIST_2D_CNN_FP/params_exp/',\
                                file_name='network_architecture.py'
                            )

    elif (dataset.lower() == 'esr'):
        if (model_to_use.lower() == 'rawaudiomultichannelcnn'):
            if not (os.path.exists('../../results/ESR_1D_CNN-transformer_FP/')):
                # Creating useful sub-folders
                os.mkdir('../../results/ESR_1D_CNN-transformer_FP/')
                os.mkdir('../../results/ESR_1D_CNN-transformer_FP/metrics/')
                os.mkdir('../../results/ESR_1D_CNN-transformer_FP/model/')
                os.mkdir('../../results/ESR_1D_CNN-transformer_FP/params_exp/')
                # Downloading files
                # Metrics an models files
                for i in range(10):
                    # Metrics
                    metric_file_to_download = 'results_exp-Exp_ESR_FullPrecision_rep-{}.pth'.format(i)
                    download_file(
                                    file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/ESR_1D_CNN-transformer_FP/metrics/{}'.format(metric_file_to_download),\
                                    dir_store='../../results/ESR_1D_CNN-transformer_FP/metrics/',\
                                    file_name=metric_file_to_download
                                )
                    # Model
                    model_file_to_download = 'final_model-Exp_ESR_FullPrecision_rep-{}'.format(i)
                    download_file(
                                    file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/ESR_1D_CNN-transformer_FP/model/{}'.format(model_file_to_download),\
                                    dir_store='../../results/ESR_1D_CNN-transformer_FP/model/',\
                                    file_name=model_file_to_download
                                )
                # Params files
                # Params of the experiment
                download_file(
                                file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/ESR_1D_CNN-transformer_FP/params_exp/params_0.pth',\
                                dir_store='../../results/ESR_1D_CNN-transformer_FP/params_exp/',\
                                file_name='params_0.pth'
                            )
                # Network architecture
                download_file(
                                file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/ESR_1D_CNN-transformer_FP/params_exp/network_architecture.py',\
                                dir_store='../../results/ESR_1D_CNN-transformer_FP/params_exp/',\
                                file_name='network_architecture.py'
                            )

        elif (model_to_use.lower() == 'timefrequency2dcnn'):
            if not (os.path.exists('../../results/ESR_2D_CNN_FP/')):
                # Creating useful sub-folders
                os.mkdir('../../results/ESR_2D_CNN_FP/')
                os.mkdir('../../results/ESR_2D_CNN_FP/metrics/')
                os.mkdir('../../results/ESR_2D_CNN_FP/model/')
                os.mkdir('../../results/ESR_2D_CNN_FP/params_exp/')
                # Downloading files
                # Metrics an models files
                for i in range(10):
                    # Metrics
                    metric_file_to_download = 'results_exp-Exp1_ClassicSpec_EEG_Binary_rep-{}.pth'.format(i)
                    download_file(
                                    file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/ESR_2D_CNN_FP/metrics/{}'.format(metric_file_to_download),\
                                    dir_store='../../results/ESR_2D_CNN_FP/metrics/',\
                                    file_name=metric_file_to_download
                                )
                    # Model
                    model_file_to_download = 'final_model-Exp1_ClassicSpec_EEG_Binary_rep-{}'.format(i)
                    download_file(
                                    file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/ESR_2D_CNN_FP/model/{}'.format(model_file_to_download),\
                                    dir_store='../../results/ESR_2D_CNN_FP/model/',\
                                    file_name=model_file_to_download
                                )
                # Params files
                # Params of the experiment
                download_file(
                                file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/ESR_2D_CNN_FP/params_exp/params_0.pth',\
                                dir_store='../../results/ESR_2D_CNN_FP/params_exp/',\
                                file_name='params_0.pth'
                            )
                # Network architecture
                download_file(
                                file_url='https://www.creatis.insa-lyon.fr/~vindas/aTTQ/results/ESR_2D_CNN_FP/params_exp/network_architecture.py',\
                                dir_store='../../results/ESR_2D_CNN_FP/params_exp/',\
                                file_name='network_architecture.py'
                            )


    # Downloading the trained auto-encoder
    # Model
    download_file(
                    file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/{}_Example_0/Model/metrics.hdf5'.format(dataset_name),\
                    dir_store='../models/{}_Example_0/Model/'.format(dataset_name),\
                    file_name='metrics.hdf5'
                )
    # Metrics
    download_file(
                    file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/{}_Example_0/Model/model.pth'.format(dataset_name),\
                    dir_store='../models/{}_Example_0/Model/'.format(dataset_name),\
                    file_name='model.pth'
                )

    # Downloading the compresed representations in the auto-encoder lantet space
    download_file(
                    file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/{}_Example_0/CompressedRepresentations/training_representations.pth'.format(dataset_name),\
                    dir_store='../models/{}_Example_0/CompressedRepresentations/'.format(dataset_name),\
                    file_name='training_representations.pth'
                )
