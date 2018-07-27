#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#                                                                             
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Enrique Corpa Rios
# DATE CREATED: 25/07/2018
# REVISED DATE:             <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print 
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir

# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Main program function defined below
def main():
    # TODO: 1. Define start_time to measure total program runtime by
    # collecting start time
    
    start_time = time() #None
    # sleep(4) # This is just a test to check the execution time
    
    # TODO: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    
    in_arg = get_input_args()
    #print('Arguments: dir= {}, model: {}, labels= {}'.format(in_arg.images, in_arg.model, in_arg.labels))
    
    # TODO: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)
    
    """
    for key in answers_dic:
        print(key + '        ' + answers_dic[key]+'\n')
    
    """
    # TODO: 4. Define classify_images() function to create the classifier 
    # labels with the classifier function uisng in_arg.arch, comparing the 
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)
    """
    print('MATCHING RESULTS: \n')
    match = 0
    for key in result_dic:
        if result_dic[key][2] == 1:
            match += 1
            print('Filename: {:<40}'.format(key) + 'Label: {:<30}'.format(result_dic[key][0]) + 'Classifier: {:<40}'.format(result_dic[key][1]))
    
    print('\nNON MATCHING RESULTS: \n')
    missmatch = 0
    for key in result_dic:
        if result_dic[key][2] == 0:
            missmatch += 1
            print('Filename: {:<40}'.format(key) + 'Label: {:<30}'.format(result_dic[key][0]) + 'Classifier: {:<40}'.format(result_dic[key][1]))
    print('\nTotal entries: {} from which {} are matches and {} are missmatches'.format(len(result_dic), match, missmatch))
    """
    
    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfiles)
    
    """
    match = 0
    missmatch = 0
    print('MATCHING RESULTS:\n')
    for key in result_dic:
        if result_dic[key][2] == 1:
            match += 1
            print('File name: {:<40} is label a dog?: {:<10} is classifier a dog?: {:<10}'.format(key, result_dic[key][3], result_dic[key][4]))
    
    print('\nMISSMATCHING RESULTS:\n')
    for key in result_dic:
        missmatch += 1
        if result_dic[key][2] == 0:
            print('File name: {:<40} is label a dog?: {:<10} is classifier a dog?: {:<10}'.format(key, result_dic[key][3], result_dic[key][4]))
    
    print('\n Total number of entries is {} from which {} are matches and {} are missmatches'.format(len(result_dic), match, missmatch))
    """
    
    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)
    
    """
    for key in results_stats_dic:
        print('Stat: {:20} = {}'.format(key,results_stats_dic[key]))
    """
    
    # TODO: 7. Define print_results() function to print summary results, 
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch, True, True)

    # TODO: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time() #None

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #None
    #tot_time = 3665
    hh = int(tot_time//3600)
    tot_time %= 3600
    mm = int(tot_time//60)
    tot_time %= 60
    ss = int(tot_time)
    
    time_msg = '{:02d}:{:02d}:{:02d}'.format(hh, mm, ss)
    print("\n** Total Elapsed Runtime:", time_msg)



# TODO: 2.-to-7. Define all the function below. Notice that the input 
# paramaters and return values have been left in the function's docstrings. 
# This is to provide guidance for acheiving a solution similar to the 
# instructor provided solution. Feel free to ignore this guidance as long as 
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object. 
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir', type = str, default = 'pet_images/', help = 'path to the folder containing the images')
    parser.add_argument('--arch', type = str, default = 'resnet', help = 'Neural Network model used for classification')
    parser.add_argument('--dogfiles', type = str, default = 'dognames.txt', help = 'Label names text file')
    
    return parser.parse_args()


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image 
    files. Reads in pet filenames and extracts the pet image labels from the 
    filenames and returns these label as petlabel_dic. This is used to check 
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)  
    """
    label_dict = dict()
    labels = []
    filename_list = listdir(image_dir)
    
    
    for filename in filename_list:
        label = ''
        tmp = filename.lower().split('_')
        for word in tmp:
            if word.isalpha():
                label += word + ' '
        label = label.strip(' ')
        if filename not in label_dict:
            label_dict[filename] = label
        else:
            print('This {} already exists in the dicionary\n'.format(filename))
            
    #print(str(len(label_dict)) + '\n' + str(label_dict))
    
    return label_dict

def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and 
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and 
                    classifer labels and 0 = no match between labels
    """
    results_dic = dict()
    
    for key in petlabel_dic: 
        image_label = classifier(images_dir + key, model)
        
        image_label = image_label.lower().strip()
        
        truth = petlabel_dic[key]
        found_idx = image_label.find(truth)
        
        if found_idx >= 0:
            if((found_idx == 0 and len(truth) == len(image_label)) or
               (((found_idx == 0) or (image_label[found_idx -1] == ' ')) and
                ((found_idx + len(truth) == len(image_label)) or
                 (image_label[found_idx + len(truth): found_idx + len(truth)+1]
                  in (',',' '))
                )
               )
            ):
                
                if key not in results_dic:
                    results_dic[key] = [truth, image_label, 1]
            else:
                if key not in results_dic:
                    results_dic[key] = [truth, image_label, 0]
            
        else:
            if key not in results_dic:
                results_dic[key] = [truth, image_label, 0]
    
    return results_dic


def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly 
    classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the 
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates 
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    dogs_names = dict()
    with open(dogsfile, 'r') as f:
        for line in f:
            if line not in dogs_names:
                dogs_names[line.strip('\n')] = {1}
            else:
                print('Warning: There is a duplicated dog name')
    
    for key in results_dic:
        if results_dic[key][0] in dogs_names : #Check if image label is dog
            results_dic[key].append(1)
        else:
            results_dic[key].append(0)
            
        if results_dic[key][1] in dogs_names: #Check if classifier label is dog
            results_dic[key].append(1)
        else:
            results_dic[key].append(0)
        #set a cero
    
    return


def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model 
    architecture on classifying images. Then puts the results statistics in a 
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
    """
    
    num_correct_dogs = 0
    num_dogs = 0
    num_correct_not_dog = 0
    num_not_dogs = 0
    num_correct_breeds = 0
    num_match = 0
    num_images = len(results_dic)
    
    for key in results_dic:
        if results_dic[key][2] == 1:
            num_match += 1  
                      
        if results_dic[key][3] == 1:
            num_dogs += 1
            if results_dic[key][2] == 1: #Mach between image label(breed if is dog) and classifier label
                num_correct_breeds += 1
            if results_dic[key][4] == 1: #Image and classifier say is a dog
                num_correct_dogs += 1  
        else:
            if results_dic[key][4] == 0:
                num_correct_not_dog += 1
    num_not_dogs = num_images - num_dogs
    
    results_stats = {'n_images': num_images,
                     'n_dogs': num_dogs,
                     'n_not_dog_img': num_not_dogs,
                     'n_correct_dogs': num_correct_dogs, 
                     'n_correct_notdogs': num_correct_not_dog,
                     'n_correct_breed': num_correct_breeds, 
                     'pct_match': num_match/num_images*100,
                     'pct_correct_dogs': num_correct_dogs/num_dogs*100,              
                     'pct_correct_breed': num_correct_breeds/num_dogs*100,
                     'pct_correct_notdogs':num_correct_not_dog/num_not_dogs*100}
    
    return results_stats


def print_results(results_dic, results_stats, model, print_incorrect_dogs = False, print_indcorrect_breed = False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """    
    print('RESULTS FOR THE CLASSIFIER WITH MODEL: {:<10}\n'.format(model))
    print('Number of images:           = {}'.format(results_stats['n_images']))
    print('Number of dog images:       = {}'.format(results_stats['n_correct_dogs']))
    print('Number of not-a dog images: = {}\n'.format(results_stats['n_correct_notdogs']))

    print('% correct dogs:             = {}'.format(results_stats['pct_correct_dogs']))
    print('% correct breed:            = {}'.format(results_stats['pct_correct_breed']))
    print('% correct not-a dog:        = {}'.format(results_stats['pct_correct_notdogs']))
    print('% match:                    = {}'.format(results_stats['pct_match']))
              
    if print_incorrect_dogs and (results_stats['n_correct_notdogs'] + results_stats['n_correct_dogs'] != results_stats['n_images']):
        print('\nPRINTING MISCLASSIFIED AS DOG LABELS:\n')
        for key in results_dic:
            if sum(results_dic[key][3:]) == 1:
                print('File = {:<30}  label: {:<20} Classifier: {:<20}'.format(key, results_dic[key][0], results_dic[key][1]))

    if print_indcorrect_breed and (results_stats['n_correct_dogs'] != results_stats['n_correct_breed']):
        print('\nPRINTING MISCLASSIFIED DOG BREEDs LABELS:\n')
        for key in results_dic:
            if sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0:
                print('File = {:<30}  label: {:<20} Classifier: {:<20}'.format(key, results_dic[key][0], results_dic[key][1]))
                
# Call to main function to run the program
if __name__ == "__main__":
    main()
