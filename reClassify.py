import os
import sys
import time
import caffe
import boto3
import cPickle
import logging
import numpy as np
import pandas as pd
import boto3.s3.transfer as tr
import json

MODEL_DEFINITION_FILE = 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED_MODEL_FILE = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
MEAN_FILE = 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
CLASS_LABELS_FILE = 'data/ilsvrc12/synset_words.txt'
BET_FILE = 'data/ilsvrc12/imagenet.bet.pickle'


class ImagenetClassifier(object):
    default_args = {'model_def_file': MODEL_DEFINITION_FILE,
                    'pretrained_model_file': PRETRAINED_MODEL_FILE,
                    'mean_file': MEAN_FILE,
                    'class_labels_file': CLASS_LABELS_FILE,
                    'bet_file': BET_FILE,
                    'gpu_mode': False,
                    'image_dim': 256,
                    'raw_scale': 255}

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.debug('Creating ImagenetClassifier instance')

        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Classifier(
                model_def_file, pretrained_model_file,
                image_dims=(image_dim, image_dim), raw_scale=raw_scale,
                mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )

        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                                         {
                                             'synset_id': l.strip().split(' ')[0],
                                             'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                                         }
                                         for l in f.readlines()
                                         ])

        self.labels = labels_df.sort('synset_id')['name'].values

        self.bet = cPickle.load(open(bet_file))

        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

        logging.debug('Finished creating Imagenet Classifier')

    def classify_images(self, images):
        try:
            logging.info('Classifying images')
            classification_results = []
            scores_list = self.net.predict(images, oversample=True)

            for scores in scores_list:
                indices = (-scores).argsort()[:5]
                predictions = self.labels[indices]

                # Compute expected information gain
                expected_infogain = np.dot(
                        self.bet['probmat'], scores[self.bet['idmapping']])
                expected_infogain *= self.bet['infogain']

                # sort the scores
                infogain_sort = expected_infogain.argsort()[::-1]

                bet_result = [self.bet['words'][v] for v in infogain_sort[:5]]
                classification_results = classification_results + [list(set(predictions) | set(bet_result))]

            logging.info('Finished classifying images')
            return True, classification_results

        except Exception as prediction_err:
            return False, prediction_err.message


class S3Wrapper(object):
    BUCKET_NAME = 'imgprocessing'
    CAFFE_FILES_FOLDER = 'CaffeFiles/'
    IMAGES_FOLDER = 'Images/'

    def __init__(self):
        logging.debug('Creating Amazon S3 transfer instance')
        self.transfer = tr.S3Transfer(boto3.client('s3'))
        logging.debug('Downloader created successfully')

    def get_caffe_files(self):
        logging.info('Downloading required files for Caffe')

        caffe_files = [
            ('deploy.prototxt', MODEL_DEFINITION_FILE),
            ('bvlc_reference_caffenet.caffemodel', PRETRAINED_MODEL_FILE),
            ('ilsvrc_2012_mean.npy', MEAN_FILE),
            ('synset_words.txt', CLASS_LABELS_FILE),
            ('imagenet.bet.pickle', BET_FILE)
        ]

        for caffe_file in caffe_files:
            logging.info('Downloading ' + caffe_file[0])
            self.transfer.download_file(self.BUCKET_NAME, self.CAFFE_FILES_FOLDER + caffe_file[0], caffe_file[1])

        logging.info('Finished downloading required files for Caffe')

    def get_images(self, images_to_download):
        logging.info('Downloading images')

        for image in images_to_download:
            logging.info('Downloading image ' + image)
            self.transfer.download_file(self.BUCKET_NAME, self.IMAGES_FOLDER + image, image)

        logging.info('Finished downloading images')


class SQSMessageReceiver(object):
    QUEUE_NAME = 'ImageProcessingQueue'
    AWS_MAX_WAIT_TIME = 20

    def __init__(self):
        logging.debug('Creating Amazon SQS Message Receiver instance')
        self.queue = boto3.resource('sqs').get_queue_by_name(QueueName=self.QUEUE_NAME)
        logging.debug('Finished creating Amazon SQS Message Receiver instance')

    def get_messages_contents(self, max_number_of_messages=1, wait_time_seconds=60 * 5):
        logging.info('Polling queue for a maximum of ' + str(wait_time_seconds) + ' seconds')

        message_contents = []
        actual_wait_time = wait_time_seconds if wait_time_seconds <= self.AWS_MAX_WAIT_TIME else self.AWS_MAX_WAIT_TIME
        wait_time_seconds -= actual_wait_time
        message_list = self.queue.receive_messages(MaxNumberOfMessages=max_number_of_messages,
                                                   WaitTimeSeconds=actual_wait_time)

        while (not message_list) and wait_time_seconds:
            actual_wait_time = (wait_time_seconds if wait_time_seconds <= self.AWS_MAX_WAIT_TIME
                                else self.AWS_MAX_WAIT_TIME)
            wait_time_seconds -= actual_wait_time
            message_list = self.queue.receive_messages(MaxNumberOfMessages=max_number_of_messages,
                                                       WaitTimeSeconds=actual_wait_time)

        for message in message_list:
            if message.body is None:
                logging.info('Message body was empty')
            else:
                logging.info('Got a message with body: ' + message.body)
                message_contents = message_contents + [message.body]

            message.delete()

        logging.info('Finished polling queue.')
        logging.info('Message contents: ' + str(message_contents))
        return message_contents


class DynamoCommunicator(object):
    CLASSES_TABLE_NAME = 'ImgProcessingClasses'
    CLASSES_TABLE_KEY = 'ClassName'
    CLASSES_ITEM_VALUE = 'Images'

    IMAGES_TABLE_NAME = 'ImgProcessingImages'
    IMAGES_TABLE_KEY = 'ImageName'
    IMAGES_ITEM_VALUE = 'Classes'

    MD5_TABLE_NAME = 'ImgProcessingMd5'
    MD5_TABLE_KEY = 'Md5'
    MD5_ITEM_VALUE = 'ImageName'

    WAIT_MULTIPLIER = 2
    MAX_WAIT_TIME = 8
    MAX_RETRIES = 5

    def __init__(self):
        logging.debug('Creating DynamoCommunicator instance')
        self.dynamodb = boto3.client('dynamodb')
        logging.debug('Finished creating DynamoCommunicator instance')

    def get_classes(self, img_names):
        logging.debug('Getting classes for images ' + str(img_names))

        result = {}
        unprocessed_items = {
            self.IMAGES_TABLE_NAME: {
                'Keys': [{self.IMAGES_TABLE_KEY: {'S': name}} for name in img_names]
            }
        }

        retries = 0
        current_wait_time = 1

        while unprocessed_items and retries < self.MAX_RETRIES:
            try:
                response = self.dynamodb.batch_get_item(RequestItems=unprocessed_items)

                for item in response['Responses'][self.IMAGES_TABLE_NAME]:
                    result[item[self.IMAGES_TABLE_KEY]['S']] = item[self.IMAGES_ITEM_VALUE]['SS']

                unprocessed_items = response['UnprocessedKeys']
                retries -= retries

                if unprocessed_items:
                    time.sleep(current_wait_time)
                    current_wait_time *= self.WAIT_MULTIPLIER

                    if current_wait_time > self.MAX_WAIT_TIME:
                        current_wait_time = self.MAX_WAIT_TIME

            except Exception as exc:
                logging.error('Caught exception while trying to get image classes. ' + exc.message)
                retries += 1
                time.sleep(current_wait_time)
                current_wait_time *= self.WAIT_MULTIPLIER

                if current_wait_time > self.MAX_WAIT_TIME:
                    current_wait_time = self.MAX_WAIT_TIME

        logging.debug('Result is ' + str(result))
        return result

    def _update_classes(self, image_name, class_list, update_string):
        classes_succeeded = set()

        for cls in class_list:
            item_updated = False
            retries = 0
            current_wait_time = 1

            while not item_updated and retries < self.MAX_RETRIES:
                try:
                    self.dynamodb.update_item(TableName=self.CLASSES_TABLE_NAME,
                                              Key={self.CLASSES_TABLE_KEY: {'S': cls}},
                                              UpdateExpression=update_string + ' :a',
                                              ExpressionAttributeValues={':a': {'SS': [image_name]}}
                                              )

                    item_updated = True
                    classes_succeeded.add(cls)

                except Exception as exc:
                    logging.error('Caught exception while trying to add image to class. ' + exc.message)
                    retries += 1
                    time.sleep(current_wait_time)
                    current_wait_time *= self.WAIT_MULTIPLIER

        logging.debug('Finished updating classes')
        return classes_succeeded

    def add_image_to_classes(self, image_name, class_list):
        logging.debug('Adding image ' + image_name + ' to classes ' + str(class_list))
        return self._update_classes(image_name, class_list, 'ADD ' + self.CLASSES_ITEM_VALUE)

    def remove_image_from_classes(self, image_name, class_list):
        logging.debug('Removing image' + image_name + ' from classes ' + str(class_list))
        return self._update_classes(image_name, class_list, 'DELETE ' + self.CLASSES_ITEM_VALUE)

    def update_images_table(self, img_names, img_classes):
        logging.debug('Updating images table')

        unprocessed_items = {
            self.IMAGES_TABLE_NAME: [{
                                         'PutRequest': {
                                             'Item': {
                                                 self.IMAGES_TABLE_KEY: {'S': name},
                                                 self.IMAGES_ITEM_VALUE: {'SS': cls}
                                             }
                                         }
                                     } for name, cls in zip(img_names, img_classes) if cls]
        }

        if not unprocessed_items[self.IMAGES_TABLE_NAME]:
            logging.info('Nothing to update.')
            return

        retries = 0
        current_wait_time = 1

        while unprocessed_items and retries < self.MAX_RETRIES:
            try:
                response = self.dynamodb.batch_write_item(RequestItems=unprocessed_items)

                unprocessed_items = response['UnprocessedItems']
                retries -= retries

                if unprocessed_items:
                    time.sleep(current_wait_time)
                    current_wait_time *= self.WAIT_MULTIPLIER

                    if current_wait_time > self.MAX_WAIT_TIME:
                        current_wait_time = self.MAX_WAIT_TIME

            except Exception as exc:
                logging.error('Caught exception while trying to update images table. ' + exc.message)
                retries += 1
                time.sleep(current_wait_time)
                current_wait_time *= self.WAIT_MULTIPLIER

                if current_wait_time > self.MAX_WAIT_TIME:
                    current_wait_time = self.MAX_WAIT_TIME

        logging.debug('Finished updating images table')

    def add_md5_and_image(self, md5, imagename):
        try:
            self.dynamodb.put_item(TableName=self.MD5_TABLE_NAME,
                                   Item = {
                                       'Md5':{
                                           'S':md5
                                       },
                                       'ImageName':{
                                           'S':imagename
                                       }
                                   }
                                )


        except Exception as exc:
            logging.error('Caught exception while trying to add md5 and imagename. ' + exc.message)

    def add_image_and_classes(self, image_name, class_list, md5):
        try:
            self.dynamodb.put_item(TableName=self.IMAGES_TABLE_NAME,
                                   Item = {
                                       'ImageName':{
                                           'S':image_name
                                       },
                                       'Classes':{
                                           'SS':class_list
                                       },
                                   }
                                )


        except Exception as exc:
            logging.error('Caught exception while trying to add image to class. ' + exc.message)

    def get_images(self, classes):
        logging.debug('Getting classes for images ' + str(classes))

        result = {}
        unprocessed_items = {
            self.CLASSES_TABLE_NAME: {
                'Keys': [{self.CLASSES_TABLE_KEY: {'S': name}} for name in classes]
            }
        }

        retries = 0
        current_wait_time = 1

        while unprocessed_items and retries < self.MAX_RETRIES:
            try:
                response = self.dynamodb.batch_get_item(RequestItems=unprocessed_items)

                for item in response['Responses'][self.CLASSES_TABLE_NAME]:
                    result[item[self.CLASSES_TABLE_KEY]['S']] = item[self.CLASSES_ITEM_VALUE]['SS']

                unprocessed_items = response['UnprocessedKeys']
                retries -= retries

                if unprocessed_items:
                    time.sleep(current_wait_time)
                    current_wait_time *= self.WAIT_MULTIPLIER

                    if current_wait_time > self.MAX_WAIT_TIME:
                        current_wait_time = self.MAX_WAIT_TIME

            except Exception as exc:
                logging.error('Caught exception while trying to get images by classes. ' + exc.message)
                retries += 1
                time.sleep(current_wait_time)
                current_wait_time *= self.WAIT_MULTIPLIER

                if current_wait_time > self.MAX_WAIT_TIME:
                    current_wait_time = self.MAX_WAIT_TIME

        logging.debug('Result is ' + str(result))
        return result

    def isStored(self, md5):
        logging.debug('Verifi if image was stored')

        try:
            response = self.dynamodb.get_item(
             TableName = self.MD5_TABLE_NAME,
             Key = {
                    self.MD5_TABLE_KEY: {'S': md5}
             }
            )

            #print json.dumps(response, indent=4, separators=(',', ': '))

            if (response['Item'][self.MD5_TABLE_KEY]):
                return True
            else: return False


        except Exception as exc:
            logging.error("Caught exception while check if md5 exist in database " + exc.message)

        return False



def should_download_caffe_files():
    for argument in sys.argv:
        if argument == '-nodownload':
            return False

    return True

