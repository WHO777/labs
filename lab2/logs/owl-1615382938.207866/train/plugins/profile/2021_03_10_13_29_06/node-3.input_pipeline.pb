  *F??????@rh???9?@2?
iIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2;?????T@!?.U{V@)?????T@1?.U{V@:Preprocessing2?
Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[0]::TFRecord<N??ot$@!????p?%@)N??ot$@1????p?%@:Advanced file read2?
ZIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl;b??m??T@!?Zi?5,V@)??iT???1?'V;????:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCache;????!?T@!???3V@)??(&o???1??f??:Preprocessing2?
rIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap<??Q???$@!??2b0&@)?QF\ ??1G??ɚ???:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch?fc%?Y??! ?o?B=??)?fc%?Y??1 ?o?B=??:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch6?Ko.??!??1??!??)6?Ko.??1??1??!??:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake>???@??!?rʴ?B??)???im??1??2??:Preprocessing2F
Iterator::Model]3?f???!??J?)????QF|?1??`?la~?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???G6W??!???Aס??)h?o}Xot?1?????u?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.