  *????F??@V??A2?
wIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::Map::MemoryCacheImpl::ParallelMapV2=l{?%??a@!?V??U?S@)l{?%??a@1?V??U?S@:Preprocessing2?
WIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::Map=F[?D?e@!N?jP?W@)؝?<?:@1?>W??!-@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::Map::MemoryCacheImpl::ParallelMapV2::FlatMap[2]::TFRecord>-?p?'P#@!u???&?@)-?p?'P#@1u???&?@:Advanced file read2?
hIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::Map::MemoryCacheImpl=1$'??a@!??,Y?S@)?Ũk????1??NA??:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::Map::MemoryCacheImpl::ParallelMapV2::FlatMap>?7?Gn?#@!R=?đ?@)????Q??1e??,?ڷ?:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch#?-?R\??!??"?ĩ??)#?-?R\??1??"?ĩ??:Preprocessing2?
dIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::Map::MemoryCache=?????a@!vW?z??S@)??N??1?̒?n??:Preprocessing2r
;Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle?/??????!v?ߝÝ??)t??%??1???????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism5s?????!V??4???)?_??9??1????W}?:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake<2V??W??!\??ϵ??)k) ????1N?#x?:Preprocessing2F
Iterator::Model+ٱ????!?s?+M???)???
~{?1}?l?n?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.