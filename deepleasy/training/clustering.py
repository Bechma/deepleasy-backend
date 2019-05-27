import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.cluster import KMeans
from tensorflow.python.keras.layers import Input, Dense, InputSpec, Layer
from tensorflow.python.keras.models import Model


def build_autoencoder(model_info):
	# build encoder

	input_layer = Input(shape=(model_info["input4encoder"],))  # input layer
	current_layer = input_layer  # init current layer with input
	for i in model_info["layers"]:  # walk through number of layers
		# add dense layer to the current layer
		current_layer = Dense(i["units"], activation=i["activation"])(current_layer)
	encoded = Dense(model_info["n_clusters"])(current_layer)  # encoded features
	encoder = Model(inputs=input_layer, outputs=encoded)  # build encoder

	# build decoder

	current_layer = encoded  # init current layer with last encoder layer
	for i in model_info["layers"][::-1]:  # walk through layers backwards
		# add dense layer to the current layer
		current_layer = Dense(i["units"], activation=i["activation"])(current_layer)
	decoded = Dense(model_info["input4encoder"])(current_layer)  # decoded features
	autoencoder = Model(inputs=input_layer, outputs=decoded)  # build auto-encoder

	# train auto-encoder

	autoencoder.compile(optimizer=model_info["autoencoderOptimizer"], loss=model_info["autoencoderLoss"])
	return autoencoder, encoder


def train_autoencoder(x_train, model_info):  # autoencoder
	autoencoder, encoder_part = build_autoencoder(model_info)

	autoencoder.fit(x_train, x_train, batch_size=model_info["batchSize"], epochs=model_info["epochs"])

	return encoder_part  # return the encoder as output


class ClusteringLayer(Layer):  # create subclass of class Layer
	def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
		super(ClusteringLayer, self).__init__(**kwargs)  # use parent class function
		self.n_clusters = n_clusters  # number of clusters
		self.alpha = alpha  # probability distribution parameter
		self.initial_weights = weights  # weights
		self.input_spec = InputSpec(ndim=2, )  # input layer dimensions

	# set the layer weights
	def build(self, input_shape):
		shape = tf.TensorShape((input_shape[1], self.n_clusters))  # kernel shape
		self.clusters = self.add_weight(
			shape=shape,
			initializer='uniform',
			name='clusters',
			trainable=True
		)  # add kernel weights
		if self.initial_weights is not None:  # if initial weights are set
			self.set_weights(self.initial_weights)  # use them
			del self.initial_weights  # delete initial weights
		self.built = True  # build the layer

	# define the forward pass
	def call(self, inputs, **kwargs):
		# to calculate soft labels we use T-distribution
		q = 1.0 / (1.0 + (K.sum(
			K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))

		q **= (self.alpha + 1.0) / 2.0  # with parameter alpha
		q = K.transpose(K.transpose(q) / K.sum(q, axis=1))  # normalize the probabilities
		return q  # return the resulting soft labels as an output

	def compute_output_shape(self, input_shape):
		return input_shape[0], self.n_clusters  # return the shape

	def get_config(self):
		config = {'n_clusters': self.n_clusters}
		base_config = super(ClusteringLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


def build_clustering_model(features, encoder, n_clusters):
	clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)

	model = Model(inputs=encoder.input, outputs=clustering_layer)  # initialize the resulting model
	model.compile(optimizer='sgd', loss='kld')  # compile the model
	kmeans = KMeans(n_clusters=n_clusters)  # initialize k-means
	labels = kmeans.fit_predict(encoder.predict(features))  # cluster encoder's outputs

	# use k-means labels as initial weights for the clustering layer
	model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
	return model
