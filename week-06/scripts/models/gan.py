#https://github.com/R-Suresh/GAN_fashion_MNIST/blob/master/gan.py
from __future__ import print_function, division

from keras.datasets import fashion_mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU

import matplotlib.pyplot as plt
import os
import sys
import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.batch_size = 128

        # image data format i schannels_last
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100        

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()     

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


        # Build the generator
        self.generator = self.build_generator()
        # [print(layer.shape) for layer in self.generator.layers]

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

  

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)
        
        print('generator output shape {}\ndiscriminator output shape {}'.format(img.shape,validity.shape))


        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def save_models(self, models_path):
        self.generator.save(os.path.join(models_path,'generator.hd5'))
        self.discriminator.save(os.path.join(models_path,'discriminator.hd5'))
        self.combined.save(os.path.join(models_path,'combined.hd5'))


    def load_models(self, models_path):
        self.generator = load_model(os.path.join(models_path,'generator.hd5'))
        self.discriminator = load_model(os.path.join(models_path,'discriminator.hd5'))
        self.combined = load_model(os.path.join(models_path,'combined.hd5'))

    def build_generator(self):

        model = Sequential()

        model.add(Dense(7 * 7 * self.batch_size, activation="relu", input_shape=(self.latent_dim,)))
        model.add(Reshape((7, 7, self.batch_size)))
        model.add(BatchNormalization(momentum=0.8))   

        model.add(UpSampling2D())
        model.add(Conv2D(1024, (3,3), strides=(3,3), padding='same', input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))        

        model.add(UpSampling2D())
        model.add(Conv2D(512, (3,3), padding ='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))        

        model.add(UpSampling2D())
        model.add(Conv2D(256, (3,3), padding ='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))    

        # model.add(UpSampling2D())
        # model.add(Conv2D(128, (3,3), padding ='same'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(LeakyReLU(alpha=0.2))    

        # model.add(UpSampling2D())
        # model.add(Conv2D(64, (3,3), padding='same'))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))    

        # model.add(UpSampling2D())
        # model.add(Conv2D(32, (3,3), padding ='same'))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(alpha=0.2))    

        model.add(Flatten())
        
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))        

        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))        

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        
        model.add(Reshape(self.img_shape))

        print('=======\nGenerator\n=======')
        model.summary()    

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        initial_bias=RandomNormal(mean=0.0, stddev=0.05, seed=None)

        model.add(Conv2D(128, (3, 3), kernel_initializer=initial_bias, strides=(3, 3), 
                        padding='same', input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
       
        print('=======\nDiscriminator\n=======')
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)


    def train(self, epochs, sample_interval=50):
        d_losses=[]
        g_losses=[]

        def plot_gan_losses(g_loss, d_loss):
            plt.plot(g_loss)
            plt.plot(d_loss)
            plt.title('GAN Loss evolution')
            plt.ylabel('')
            plt.xlabel('epoch')
            plt.legend(['Generator', 'Discriminator'], loc='best')
            plt.show()

        # Load the dataset
        (X_train, _), (_, _) = fashion_mnist.load_data()


        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            if epoch == 0:
                print('random image shape {}\ngenerated image shape {}'.format(imgs.shape, gen_imgs.shape))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
           

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            g_losses.append(g_loss)
            d_losses.append(d_loss)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # Plot the progress
                print('===EPOCH {}==='.format(epoch))
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                self.sample_images(epoch)

        plot_gan_losses(g_losses, d_losses)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        #fig.savefig("images/%d.png" % epoch)
        plt.show()
        plt.close()