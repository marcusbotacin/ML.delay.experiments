# Keep detection statistics 
class Stats
	# add new detection results to the statistics
	# called to add data every day
	def add()
	# re-evaluate statistics
	# this is important because classifiers might be updated any time
	def rerun()
	# print the total stats
	# called in the end
	# might generate a graph or so
	def print()

# host the analysis queue
class Queue
	# add more samples to the analysis queue
	def add
	# remove the last samples from the analysis queue if they are ready
	def get_ready

# This is the pipeline itself
class Tester
	# You supply a stream of malware samples
	# A stream of goodware samples
	# A trained (initial) ML model
	# the confidence to require a sample go to oracle
	# the size of the oracle queue and its size
	# The class keeps a list of statistics
	def __init__(self, mw_dataset, gw_dataset, model, confidence, queue_size, analysis_time):
		self.mw_dataset = mw_dataset
		self.gw_dataset = gw_dataset
		self.model = model
		self.confidence = confidence
		self.queue = Queue(queue_size, analysis_time)
		self.triage_stats = Stats()
		self.update_stats = Stats()
		self.oracle_stats = Stats()	

	# given an entire dataset, return batches with a given number of malware and goodware for every day
	# dataset must be temporally ordered to be a stream
	def generate_batch(mw_day, gw_day):
		return mw_batches, gw_batches

	# select the samples that should be sent to a secondary analysis queue based on the confidence score
	def get_queue_sample(self,mw_batch,gw_batch,mw_labels,mw_confidence,gw_labels,gw_confidence):
		# if self.threshold_confidence
		return mw_batch, gw_batch

	# run the pipeline
	def test(self):
		# follow the stream of samples
		for day, mw_batch, gw_batch in enumerate(self.generate_batch(mw_fraction)):
			# perform triage
			mw_labels, mw_confidence, gw_labels, gw_confidence = 
				self.model.triage(mw_batch,gw_batch)
			# here we have triage results, add to the statistics
			self.triage_stats.add(...)
			# send labels to retrain
			# if model does not have retrain (naive pipeline), invokes None
			self.model.update(
				mw_batch,gw_batch,
				mw_labels,mw_confidence,
				gw_labels,gw_confidence)
			# here we might have a new classifier deployed, we need to re-run statistics to accoutn the new classification labels.
			self.update_stats.rerun(...)
			# select a fraction of samples to go on queue
			# not all can go because queue is limited size
			# selecting based on confidence score
			mw_batch_queue, gw_batch_queue = self.get_queue_samples(
				mw_batch,gw_batch,
				mw_labels,mw_confidence,
				gw_labels,gw_confidence)
			# add them to analysis queue
			# their analysis will be ready N days ahead
				self.queue.add(mw_batch_queue,gw_batch_queue, day)
			# remove samples whose analysis is ready in this day/hour
			# a variable number of samples will be here
			# their confidence is 100%, oracle never makes mistakes
				mw_queue_labels, gw_queue_labels = self.queue.get_ready(day)
			# The ready are used to update the model
			self.model.update(
				mw_batch,gw_batch,
				mw_labels,mw_confidence,
				gw_labels,gw_confidence)
			# rerun stats again
			self.oracle_stats.rerun(...)

	# after the days have passed, we can see what happened
	self.triage_stats.print()
	self.update_stats.print()
	self.oracle_stats.print()

# create some models, externally trained
models[i] = Model()
models[i].updated = Updates() # Each model might have a distinct update mechanism or none in case of simple triage
models[i].drift_detector = Detector() # Each model might have a distinct drift detector

# Simulation
# vary the models
for model in models:
  # vary the mw datasets
  for mw_dataset in mw_datasets:
    # vary the gw datasets
    for gw_dataset in gw_datasets:
      # vary the confidence level
      for c_level in range(confidence_levels):
        # vary queue size
	for q_size in range(queue_sizes):
          # vary analysis length
	  for a_time in range(analysis_time):
		# simulate this scenario
		t = Tester(mw_dataset,gw_dataset,model,c_level,q_size,a_time)
		res = t.test()
