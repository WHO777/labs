  *???????@?????@2?
XIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2M/??wHU@!???B?U@)/??wHU@1???B?U@:Preprocessing2?
nIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMap[0]::TFRecordM?[ AI(@!fIٝ??(@)?[ AI(@1fIٝ??(@:Advanced file read2?
aIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMapMK>v(?(@!?l??-)@)D??{???1?HDd??:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch?|???ݡ?!(N?%U??)?|???ݡ?1(N?%U??:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch?? ??ԅ?!?a??f??)?? ??ԅ?1?a??f??:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTakeA*Ŏơ??!??,9??)J?????1??su???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?j??P???!@}?$?G??)??Rp?1?]?Ïwp?:Preprocessing2F
Iterator::Model{j??U???!?_~t2??)f?O7P?m?1?+????n?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.