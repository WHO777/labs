"?
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1?x????5AA?x????5Aa"_?K???i"_?K????Unknown
bHost
DecodeJpeg"
DecodeJpeg(X1??k?A9?D2N?P?@A??k?AI?D2N?P?@ad???"??i??;????Unknown
dHostCast"convert_image/Cast(V1{?G *A9y???1?@A{?G *AIy???1?@a_???w???igkm:ml???Unknown
?HostDataset"rIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2(X1q=
??<?@9?CM??Z?@Aq=
??<?@I?CM??Z?@a]?h?:??i????`????Unknown
^HostMul"convert_image(W1V?H?@9??lj?P?@AV?H?@I??lj?P?@aл
?F!??iJ?dVu????Unknown
qHostResizeBilinear"resize/ResizeBilinear(W1=
ף`n?@9},z?{@A=
ף`n?@I},z?{@a????e??i^?LVd???Unknown
?HostDataset"?Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[2]::TFRecord(Y1??K7i??@97o?'@de@A??K7i??@I7o?'@de@a??2??Vy?iucS??????Unknown
?	HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(Y1??? P??@9щ?҃R@A??? P??@Iщ?҃R@ar{?7R?e?i??稬???Unknown
n
HostSub"per_image_standardization/sub(Y1=
ףs?@9?ᔒ??Q@A=
ףs?@I?ᔒ??Q@a?ZR?d?iK?9|????Unknown
?HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(Y1F????D?@9?᷅ysQ@AF????D?@I?᷅ysQ@ay?,??d?iR???'????Unknown
pHostMean"per_image_standardization/Mean(Y1??x?h?@9AQ????P@A??x?h?@IAQ????P@a?r?#??c?iů??????Unknown
nHostRealDiv"per_image_standardization(Y1?n?@4?@9?A^ ?C@A?n?@4?@I?A^ ?C@a?^?I?+W?i?}?έ????Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(Y1L7?A?՗@9??Ϫ?#1@AL7?A?՗@I??Ϫ?#1@a?*?8nMD?i??*?????Unknown
[HostOneHot"one_hot(Y1d;?O???@9ؿ???@Ad;?O???@Iؿ???@a?F?ꖒ/?i?RS?????Unknown
?HostDataset"cIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl(X1H?zY?@9Lu?TWo?@A
ףp=4|@I?V1i?@a?J??(?i????:????Unknown
?HostDataset"{Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap(Y1??C?\?@9?Ƀ??e@AF?????a@Iۯ?????a???????i??0F?????Unknown
?HostDataset"_Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCache(X1??v??`?@9??'?t?@A-?????^@I??????a?y?{?X
?i}?ީ????Unknown
?HostDataset"EIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch(1B`??"?[@9B`??"?[@AB`??"?[@IB`??"?[@a?G2n??iFݖ?|????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1)\????C@9)\????C@A)\????C@I)\????C@a4?5???>i???l?????Unknown?
eHost
LogicalAnd"
LogicalAnd(1ףp=
w=@9ףp=
w=@Aףp=
w=@Iףp=
w=@a?ޔ???>iG????????Unknown?
iHostWriteSummary"WriteSummary(1B`??"?5@9B`??"?5@AB`??"?5@IB`??"?5@a?? ???>ih??>?????Unknown?
lHostIteratorGetNext"IteratorGetNext(1{?G??/@9{?G??/@A{?G??/@I{?G??/@a?b=?>i?|??????Unknown
?HostDataset";Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle(1??? ?:_@9??? ?:_@A#??~j?,@I#??~j?,@a??uz?>i?I~?????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1?S㥛la@9?S㥛la@A??/??!@I??/??!@a???{?>i?m??????Unknown
eHost_Send"IteratorGetNext/_3(1??K7?A@9??K7?A@A??K7?A@I??K7?A@ao???u??>i??*?????Unknown
dHostDataset"Iterator::Model(1
ףp=.b@9
ףp=.b@ANbX94@INbX94@a%C?<˝?>i??=R?????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1????MN`@9????MN`@A??Q?@I??Q?@a\ԃ:^??>ic?????Unknown
eHost_Send"IteratorGetNext/_1(15^?I	@95^?I	@A5^?I	@I5^?I	@as1aM?>i縱?????Unknown
?Host	_HostSend"Ecategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_5(1?v??/??9?v??/??A?v??/??I?v??/??a???ۘ>i???x?????Unknown
aHostIdentity"Identity(1?Zd;???9?Zd;???A?Zd;???I?Zd;???a?7j.??>i      ???Unknown?*?
bHost
DecodeJpeg"
DecodeJpeg(X1??k?A9?D2N?P?@A??k?AI?D2N?P?@a?fd?$??i?fd?$???Unknown
dHostCast"convert_image/Cast(V1{?G *A9y???1?@A{?G *AIy???1?@amI?㾠??i?? 4z???Unknown
?HostDataset"rIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2(X1q=
??<?@9?CM??Z?@Aq=
??<?@I?CM??Z?@a?Sg????i??խ???Unknown
^HostMul"convert_image(W1V?H?@9??lj?P?@AV?H?@I??lj?P?@a3S?-???i~???a???Unknown
qHostResizeBilinear"resize/ResizeBilinear(W1=
ף`n?@9},z?{@A=
ף`n?@I},z?{@ak?????i/=?`????Unknown
?HostDataset"?Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[2]::TFRecord(Y1??K7i??@97o?'@de@A??K7i??@I7o?'@de@a? ??0??i?]9?#???Unknown
?HostSquare";per_image_standardization/reduce_std/reduce_variance/Square(Y1??? P??@9щ?҃R@A??? P??@Iщ?҃R@a??k!z?i٭J?f9???Unknown
nHostSub"per_image_standardization/sub(Y1=
ףs?@9?ᔒ??Q@A=
ףs?@I?ᔒ??Q@a~?w?9?x?i??sCk???Unknown
?	HostParseExampleV2".ParseSingleExample/ParseExample/ParseExampleV2(Y1F????D?@9?᷅ysQ@AF????D?@I?᷅ysQ@aKN??x?i?9]I????Unknown
p
HostMean"per_image_standardization/Mean(Y1??x?h?@9AQ????P@A??x?h?@IAQ????P@aE?#[6?w?iv????????Unknown
nHostRealDiv"per_image_standardization(Y1?n?@4?@9?A^ ?C@A?n?@4?@I?A^ ?C@a???m??k?iD:?g????Unknown
?HostMean";per_image_standardization/reduce_std/reduce_variance/Mean_1(Y1L7?A?՗@9??Ϫ?#1@AL7?A?՗@I??Ϫ?#1@aP@B?0X?i:M[?????Unknown
[HostOneHot"one_hot(Y1d;?O???@9ؿ???@Ad;?O???@Iؿ???@a?fC+?B?i;?3????Unknown
?HostDataset"cIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl(X1H?zY?@9Lu?TWo?@A
ףp=4|@I?V1i?@a(?????<?i????????Unknown
?HostDataset"{Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap(Y1??C?\?@9?Ƀ??e@AF?????a@Iۯ?????a?O,?^A"?iJ????????Unknown
?HostDataset"_Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCache(X1??v??`?@9??'?t?@A-?????^@I??????a?5l?d?i?G??????Unknown
?HostDataset"EIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch(1B`??"?[@9B`??"?[@AB`??"?[@IB`??"?[@aE?<q?i?'mj?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1)\????C@9)\????C@A)\????C@I)\????C@a????i |?z????Unknown?
eHost
LogicalAnd"
LogicalAnd(1ףp=
w=@9ףp=
w=@Aףp=
w=@Iףp=
w=@a?-ӃO??>i??{IS????Unknown?
iHostWriteSummary"WriteSummary(1B`??"?5@9B`??"?5@AB`??"?5@IB`??"?5@a?d??N?>i?WC?????Unknown?
lHostIteratorGetNext"IteratorGetNext(1{?G??/@9{?G??/@A{?G??/@I{?G??/@a1???]:?>i??[?????Unknown
?HostDataset";Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle(1??? ?:_@9??? ?:_@A#??~j?,@I#??~j?,@a(?A??)?>i?煽????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1?S㥛la@9?S㥛la@A??/??!@I??/??!@a"?)?>ix???????Unknown
eHost_Send"IteratorGetNext/_3(1??K7?A@9??K7?A@A??K7?A@I??K7?A@a?N?Ҵ?>i?\	?????Unknown
dHostDataset"Iterator::Model(1
ףp=.b@9
ףp=.b@ANbX94@INbX94@a^?4?r??>i???Q?????Unknown
?HostDataset"2Iterator::Model::MaxIntraOpParallelism::FiniteTake(1????MN`@9????MN`@A??Q?@I??Q?@aE!s?>i,l??????Unknown
eHost_Send"IteratorGetNext/_1(15^?I	@95^?I	@A5^?I	@I5^?I	@a.Ya	Va?>i??k??????Unknown
?Host	_HostSend"Ecategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_5(1?v??/??9?v??/??A?v??/??I?v??/??a???M??>iU?P??????Unknown
aHostIdentity"Identity(1?Zd;???9?Zd;???A?Zd;???I?Zd;???a??J?*?>i     ???Unknown?2GPU