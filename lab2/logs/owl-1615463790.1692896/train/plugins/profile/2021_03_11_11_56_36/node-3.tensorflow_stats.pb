"? 
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
bHost
DecodeJpeg"
DecodeJpeg(1?G?Բ?@9??Q?Χ?@A?G?Բ?@I??Q?Χ?@aqM?M???iqM?M????Unknown
dHostCast"convert_image/Cast(1?"??>??@9??y????@A?"??>??@I??y????@a???! ??i??g????Unknown
BHostIDLE"IDLE1B`??*??@AB`??*??@a?,??-s??i2x?h2????Unknown
^HostMul"convert_image(
1??Q?~?@9?????҄@A??Q?~?@I?????҄@aJ?r ??i????9????Unknown
qHostResizeBilinear"resize/ResizeBilinear(
1?p=
W??@9?&1?.?@A?p=
W??@I?&1?.?@a???@???i?JH??????Unknown
?HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(1?z????@9?z???h@A?z????@I?z???h@a??J????i^?*?db???Unknown
nHostSub"per_image_standardization/sub(1X9???@9X9??c@AX9???@IX9??c@a?;?ZK??i}???????Unknown
?	HostDataset"Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[0]::TFRecord(1Zd;ߏߡ@9Zd;ߏ?a@AZd;ߏߡ@IZd;ߏ?a@a?tn.⸈?iOF9}u.???Unknown
p
HostMean"per_image_standardization/Mean(1?I???@9?I??R@A?I???@I?I??R@a??9??z?i????b???Unknown
?HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(1?~j??7?@9?~j??7R@A?~j??7?@I?~j??7R@a??:B&3y?i?/0?????Unknown
nHostRealDiv"per_image_standardization(1m????2?@9m????2L@Am????2?@Im????2L@azfqP??s?i???????Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(1'1?~v@9'1?~6@A'1?~v@I'1?~6@a????_?i??I?????Unknown
eHost_Send"IteratorGetNext/_3(1`??"??k@9`??"??k@A`??"??k@I`??"??k@aD#z`?PS?i$??L????Unknown
?HostDataset"iIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2(1L7?A`%h@9L7?A`%(@AL7?A`%h@IL7?A`%(@au?L?P?i? !4?????Unknown
?HostDataset"ZIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl(1?t?Fv@9?t?F6@A-????fd@I-????f$@a?=?l?7L?i?C<0?????Unknown
eHost_Send"IteratorGetNext/_1(1??????`@9??????`@A??????`@I??????`@arș??=G?iX?z??????Unknown
[HostOneHot"one_hot(1+??Η^@9+??Η@A+??Η^@I+??Η@a?,fs(E?i??>??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1j?t?tP@9j?t?tP@Aj?t?tP@Ij?t?tP@a????#?6?i????????Unknown?
eHost
LogicalAnd"
LogicalAnd(1?S㥛H@9?S㥛H@A?S㥛H@I?S㥛H@a???k?0?ir&+??????Unknown?
?HostDataset"rIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap(1}?5^?5?@9}?5^?5b@A??v???E@I??v???@a?n????-?iA÷?????Unknown
iHostWriteSummary"WriteSummary(1NbX9B@9NbX9B@ANbX9B@INbX9B@a??[??)?iֶ]?'????Unknown?
?HostDataset"<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch(1^?I?>@9^?I?>@A^?I?>@I^?I?>@a???d?5%?i?/{????Unknown
vHostMaximum"!per_image_standardization/Maximum(1
ףp=J<@9
ףp=J??A
ףp=J<@I
ףp=J??a???ִ?#?i!uQ:?????Unknown
?HostDataset"IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch(1Zd;?O?;@9Zd;?O?;@AZd;?O?;@IZd;?O?;@a?kDd#?ih??????Unknown
?HostDataset"VIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCache(1d;?O??w@9d;?O??7@Aj?t??8@Ij?t????a?:HM?!?i??F?????Unknown
{HostSqrt")per_image_standardization/reduce_std/Sqrt(1+??n6@9+??n??A+??n6@I+??n??a?@?w?i??ā?????Unknown
lHostIteratorGetNext"IteratorGetNext(17?A`?P-@97?A`?P-@A7?A`?P-@I7?A`?P-@aj??[F?i?{???????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(17?A`??I@97?A`??I@AˡE???$@IˡE???$@a+?س???i?Jaw????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1? ?rhQD@9? ?rhQD@AV-???#@IV-???#@aN?=?H??i???p????Unknown
dHostDataset"Iterator::Model(1D?l???M@9D?l???M@A1?Zd!@I1?Zd!@a??)?u?i?l[??????Unknown
? Host	_HostSend"Ecategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_5(1㥛? ?@9㥛? ?@A㥛? ?@I㥛? ?@aLL?&??>iA?O$?????Unknown
a!HostIdentity"Identity(1d;?O????9d;?O????Ad;?O????Id;?O????aI?{?`??>i?????????Unknown?*?
bHost
DecodeJpeg"
DecodeJpeg(1?G?Բ?@9??Q?Χ?@A?G?Բ?@I??Q?Χ?@a?&?0????i?&?0?????Unknown
dHostCast"convert_image/Cast(1?"??>??@9??y????@A?"??>??@I??y????@a?O?C???i&?
?????Unknown
^HostMul"convert_image(
1??Q?~?@9?????҄@A??Q?~?@I?????҄@a\????@??i???T????Unknown
qHostResizeBilinear"resize/ResizeBilinear(
1?p=
W??@9?&1?.?@A?p=
W??@I?&1?.?@a4X?????i/?<d?W???Unknown
?HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(1?z????@9?z???h@A?z????@I?z???h@a8??%?$??ij?? ???Unknown
nHostSub"per_image_standardization/sub(1X9???@9X9??c@AX9???@IX9??c@a??^@??i?R???????Unknown
?HostDataset"Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[0]::TFRecord(1Zd;ߏߡ@9Zd;ߏ?a@AZd;ߏߡ@IZd;ߏ?a@a??N9=???iQ????????Unknown
pHostMean"per_image_standardization/Mean(1?I???@9?I??R@A?I???@I?I??R@a????j#??i0AvC?=???Unknown
?	HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(1?~j??7?@9?~j??7R@A?~j??7?@I?~j??7R@a??\?_&?ir??{???Unknown
n
HostRealDiv"per_image_standardization(1m????2?@9m????2L@Am????2?@Im????2L@a?Hc2nx?i?g?????Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(1'1?~v@9'1?~6@A'1?~v@I'1?~6@aqU?:c?iuirG????Unknown
eHost_Send"IteratorGetNext/_3(1`??"??k@9`??"??k@A`??"??k@I`??"??k@ag?^?W?i?u?7????Unknown
?HostDataset"iIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2(1L7?A`%h@9L7?A`%(@AL7?A`%h@IL7?A`%(@a?"??k?T?i:?7׉????Unknown
?HostDataset"ZIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl(1?t?Fv@9?t?F6@A-????fd@I-????f$@a??%??pQ?i,???B????Unknown
eHost_Send"IteratorGetNext/_1(1??????`@9??????`@A??????`@I??????`@a=???ҺL?i??R?p????Unknown
[HostOneHot"one_hot(1+??Η^@9+??Η@A+??Η^@I+??Η@a?$?4]'J?i?3???????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1j?t?tP@9j?t?tP@Aj?t?tP@Ij?t?tP@a??z?!<?i???????Unknown?
eHost
LogicalAnd"
LogicalAnd(1?S㥛H@9?S㥛H@A?S㥛H@I?S㥛H@a??K??4?irL??????Unknown?
?HostDataset"rIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap(1}?5^?5?@9}?5^?5b@A??v???E@I??v???@a@???jj2?id?_????Unknown
iHostWriteSummary"WriteSummary(1NbX9B@9NbX9B@ANbX9B@INbX9B@a?wn?G?.?i?:w?M????Unknown?
?HostDataset"<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch(1^?I?>@9^?I?>@A^?I?>@I^?I?>@aLK.?7*?i??%?????Unknown
vHostMaximum"!per_image_standardization/Maximum(1
ףp=J<@9
ףp=J??A
ףp=J<@I
ףp=J??aV???X/(?i:?st????Unknown
?HostDataset"IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch(1Zd;?O?;@9Zd;?O?;@AZd;?O?;@IZd;?O?;@a?f?*Ս'?i?G???????Unknown
?HostDataset"VIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCache(1d;?O??w@9d;?O??7@Aj?t??8@Ij?t????a??]?%?i?"<g=????Unknown
{HostSqrt")per_image_standardization/reduce_std/Sqrt(1+??n6@9+??n??A+??n6@I+??n??a???o?-#?i???p????Unknown
lHostIteratorGetNext"IteratorGetNext(17?A`?P-@97?A`?P-@A7?A`?P-@I7?A`?P-@a?Ţ???i῾8????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(17?A`??I@97?A`??I@AˡE???$@IˡE???$@aخ??q??iDOO??????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1? ?rhQD@9? ?rhQD@AV-???#@IV-???#@a????ik??N????Unknown
dHostDataset"Iterator::Model(1D?l???M@9D?l???M@A1?Zd!@I1?Zd!@a?A?????i???????Unknown
?Host	_HostSend"Ecategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_5(1㥛? ?@9㥛? ?@A㥛? ?@I㥛? ?@aD???	?>i???????Unknown
aHostIdentity"Identity(1d;?O????9d;?O????Ad;?O????Id;?O????a???????>i      ???Unknown?2GPU