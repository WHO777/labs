  *^?Ib?@??Mbܡ?@2?
XIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2+?=~?M@!T?y"??T@)?=~?M@1T?y"??T@:Preprocessing2?
nIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMap[0]::TFRecord 5y?j??&@!?A??!0@)5y?j??&@1?A??!0@:Advanced file read2?
aIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMap Q}>ʸ&@!J?ST??0@)?M????1F?]??:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatchRD?U????!?o?}? ??)RD?U????1?o?}? ??:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch?
}????!????Ɣ?)?
}????1????Ɣ?:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake??eO???!??q u
??)?M???Pt?1t? "?}?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism[? m?Y??!????ʲ??)????H?m?1?Ww4?lu?:Preprocessing2F
Iterator::Modell??[??!?^M\Q??)???@?Mm?16},???t?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.