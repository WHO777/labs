  *	???S???@2|
EIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2?E?n?q)@!??????X@)?]??NG)@1?~UH??X@:Preprocessing2?
RIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCache@b??m?R??!H??X???)?8??m4??1??{?^???:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl@=?q??!?W??R??)=?q??1?W??R??:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteTake::PrefetchS=??M??!??ڊ8ȹ?)S=??M??1??ڊ8ȹ?:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake??b????!??Ӫ7??)?<?E~???1l?B???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???-=???!A?rc???)???+?s?1֢?AJm??:Preprocessing2F
Iterator::ModelL????q??!I?Ĕ???)|E?^s?1 H?<???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.