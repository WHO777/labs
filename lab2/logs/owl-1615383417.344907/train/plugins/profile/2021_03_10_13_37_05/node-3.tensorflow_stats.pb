"?
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1ףp??{AAAףp??{AAa??????i???????Unknown
bHost
DecodeJpeg"
DecodeJpeg(;1-?C?A9?2??@A-?C?AI?2??@aj?J)????i?^??????Unknown
dHostCast"convert_image/Cast(:1?v??̑
A9狋oQ?@A?v??̑
AI狋oQ?@a?NЇ????iкXO????Unknown
?HostDataset"iIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2(;1d;?O'??@9E??c??@Ad;?O'??@IE??c??@a0??\??i????????Unknown
^HostMul"convert_image(:1㥛Č??@9??Bŏ@A㥛Č??@I??Bŏ@aw?3??i???i?e???Unknown
qHostResizeBilinear"resize/ResizeBilinear(:1?VN??@9?a??D?@A?VN??@I?a??D?@ao9^4T??i?o]?V????Unknown
?HostDataset"Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[0]::TFRecord(<1????k??@9?T??Pc@A????k??@I?T??Pc@a???l??g?i(U?>?????Unknown
?	HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(;1㥛Ġl?@9??嶥?R@A㥛Ġl?@I??嶥?R@a=❝ W?iGF??l????Unknown
p
HostMean"per_image_standardization/Mean(;1{?GA??@9QdjI?R@A{?GA??@IQdjI?R@a6$R&??U?iYo,?c????Unknown
?HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(<1d;?O???@97???Q@Ad;?O???@I7???Q@a??1?I?U?i"?Y????Unknown
nHostSub"per_image_standardization/sub(;1^?I?@9?6#?|I@A^?I?@I?6#?|I@a??O?i??????Unknown
nHostRealDiv"per_image_standardization(;1}?5^???@9O???:SG@A}?5^???@IO???:SG@a?%??bL?ig	G?3????Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(;1+????@9?????)2@A+????@I?????)2@a?W?(?6?i?#???????Unknown
[HostOneHot"one_hot(<1H?z?G{@9???7B@AH?z?G{@I???7B@aCJZ'?"?iw???????Unknown
?HostDataset"ZIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl(;1P??nX?@9???P?@A??Q?1w@I?k	D(@a?'??ǝ?i??
?????Unknown
?HostDataset"VIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCache(;1sh????@9????Jɕ@AD?l???X@I'??*????aѶ???l ?i??y?M????Unknown
?HostDataset"rIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap(<1??~j<?@9???bBc@A???QhX@I<?'????a???XM ?i;??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff?B@9fffff?B@Afffff?B@Ifffff?B@aҐ????>i???զ????Unknown?
?HostDataset"<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch(1q=
ףp:@9q=
ףp:@Aq=
ףp:@Iq=
ףp:@aJ??_?s?>i???I?????Unknown
eHost
LogicalAnd"
LogicalAnd(1=
ףp?7@9=
ףp?7@A=
ףp?7@I=
ףp?7@aF'5?C??>i	?$
?????Unknown?
iHostWriteSummary"WriteSummary(1?K7?A?5@9?K7?A?5@A?K7?A?5@I?K7?A?5@a??U?Զ?>i??e?????Unknown?
?HostDataset"IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch(1Zd;?O?3@9Zd;?O?3@AZd;?O?3@IZd;?O?3@a}?? gN?>i??????Unknown
?Host	_HostSend"3gradient_tape/categorical_crossentropy/mul/Shape/_5(1ףp=
W%@9ףp=
W%@Aףp=
W%@Iףp=
W%@a??o?+?>i?D???????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1??/?$B@9??/?$B@AV-?#@IV-?#@a??>?U ?>ig???????Unknown
dHostDataset"Iterator::Model(1sh??|H@9sh??|H@A?Zd?@I?Zd?@ax?????>i???V?????Unknown
lHostIteratorGetNext"IteratorGetNext(1??S㥛@9??S㥛@A??S㥛@I??S㥛@a?q?.>?>i~?ff?????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1??Mb?D@9??Mb?D@A??????@I??????@a2j?̃?>i?"???????Unknown
eHost_Send"IteratorGetNext/_3(1+???@9+???@A+???@I+???@a%7?D n?>i?&???????Unknown
eHost_Send"IteratorGetNext/_1(1?x?&1??9?x?&1??A?x?&1??I?x?&1??a"%??*?>i???????Unknown
aHostIdentity"Identity(1?Zd;???9?Zd;???A?Zd;???I?Zd;???a.v??;?>i     ???Unknown?*?
bHost
DecodeJpeg"
DecodeJpeg(;1-?C?A9?2??@A-?C?AI?2??@a???7????i???7?????Unknown
dHostCast"convert_image/Cast(:1?v??̑
A9狋oQ?@A?v??̑
AI狋oQ?@awN?sv??iB?t?&????Unknown
?HostDataset"iIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2(;1d;?O'??@9E??c??@Ad;?O'??@IE??c??@a????????i[?+۵???Unknown
^HostMul"convert_image(:1㥛Č??@9??Bŏ@A㥛Č??@I??Bŏ@a)
????i R'EZ????Unknown
qHostResizeBilinear"resize/ResizeBilinear(:1?VN??@9?a??D?@A?VN??@I?a??D?@a3)?O???i?T&?Y????Unknown
?HostDataset"Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[0]::TFRecord(<1????k??@9?T??Pc@A????k??@I?T??Pc@a?_Ӯ'??i2?rU?=???Unknown
?HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(;1㥛Ġl?@9??嶥?R@A㥛Ġl?@I??嶥?R@a&??t?i~ؓ?<g???Unknown
pHostMean"per_image_standardization/Mean(;1{?GA??@9QdjI?R@A{?GA??@IQdjI?R@a????s?i??????Unknown
?	HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(<1d;?O???@97???Q@Ad;?O???@I7???Q@a??,WZ?s?i?_?a?????Unknown
n
HostSub"per_image_standardization/sub(;1^?I?@9?6#?|I@A^?I?@I?6#?|I@a 6T??k?ik$?????Unknown
nHostRealDiv"per_image_standardization(;1}?5^???@9O???:SG@A}?5^???@IO???:SG@aJ?J??vi?i?? ?1????Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(;1+????@9?????)2@A+????@I?????)2@aJK???S?iD?B?????Unknown
[HostOneHot"one_hot(<1H?z?G{@9???7B@AH?z?G{@I???7B@aѼ??&@?is	?b%????Unknown
?HostDataset"ZIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl(;1P??nX?@9???P?@A??Q?1w@I?k	D(@aPh'??v;?i`N?3?????Unknown
?HostDataset"VIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCache(;1sh????@9????Jɕ@AD?l???X@I'??*????a%???w?i??m?????Unknown
?HostDataset"rIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap(<1??~j<?@9???bBc@A???QhX@I<?'????a?>??	??ihY?$g????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff?B@9fffff?B@Afffff?B@Ifffff?B@a????;?i?]?????Unknown?
?HostDataset"<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch(1q=
ףp:@9q=
ףp:@Aq=
ףp:@Iq=
ףp:@aɏ??.O?>iAa??????Unknown
eHost
LogicalAnd"
LogicalAnd(1=
ףp?7@9=
ףp?7@A=
ףp?7@I=
ףp?7@a׼??B?>i?L?67????Unknown?
iHostWriteSummary"WriteSummary(1?K7?A?5@9?K7?A?5@A?K7?A?5@I?K7?A?5@a???1???>i???j????Unknown?
?HostDataset"IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch(1Zd;?O?3@9Zd;?O?3@AZd;?O?3@IZd;?O?3@a#???ʘ?>i
????????Unknown
?Host	_HostSend"3gradient_tape/categorical_crossentropy/mul/Shape/_5(1ףp=
W%@9ףp=
W%@Aףp=
W%@Iףp=
W%@a?q_#E?>i&??0?????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1??/?$B@9??/?$B@AV-?#@IV-?#@a;f_?R?>i???????Unknown
dHostDataset"Iterator::Model(1sh??|H@9sh??|H@A?Zd?@I?Zd?@aP??P?x?>i?)ƿ?????Unknown
lHostIteratorGetNext"IteratorGetNext(1??S㥛@9??S㥛@A??S㥛@I??S㥛@adR???#?>is?Q?????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1??Mb?D@9??Mb?D@A??????@I??????@a4?????>i?x??????Unknown
eHost_Send"IteratorGetNext/_3(1+???@9+???@A+???@I+???@a4կ?M1?>i???'?????Unknown
eHost_Send"IteratorGetNext/_1(1?x?&1??9?x?&1??A?x?&1??I?x?&1??aq???	??>i?~??????Unknown
aHostIdentity"Identity(1?Zd;???9?Zd;???A?Zd;???I?Zd;???a??????>i      ???Unknown?2GPU