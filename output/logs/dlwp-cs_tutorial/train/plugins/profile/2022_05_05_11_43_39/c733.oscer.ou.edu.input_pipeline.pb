	X?2?=??@X?2?=??@!X?2?=??@	?k???????k??????!?k??????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$X?2?=??@衶???A???Ho?@Y?V?S?M@*	??????@2t
=Iterator::Model::MaxIntraOpParallelism::FlatMap[0]::Generator???ÈM@!8????X@)???ÈM@18????X@:Preprocessing2F
Iterator::Model?3ڪ$?M@!      Y@)Uܸ???p?1???|?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???M@!nЋӍ?X@)?KU??o?1?~L8(Sz?:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::FlatMap??hW!?M@!;???$?X@){/?h?g?1? ????s?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?k??????I)?:??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	衶???衶???!衶???      ??!       "      ??!       *      ??!       2	???Ho?@???Ho?@!???Ho?@:      ??!       B      ??!       J	?V?S?M@?V?S?M@!?V?S?M@R      ??!       Z	?V?S?M@?V?S?M@!?V?S?M@b      ??!       JCPU_ONLYY?k??????b q)?:??X@