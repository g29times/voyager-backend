from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")

# TODO 与Test结论有不同 调查原因 2024-12-12 milvus_demo的代码效果更好
# DOC https://milvus.io/api-reference/pymilvus/v2.4.x/EmbeddingModels/VoyageEmbeddingFunction/VoyageEmbeddingFunction.md
# 元春和迎春的关系如何
# Jina work 贾元春 贾迎春 贾探春 贾宝玉
from pymilvus.model.dense import JinaEmbeddingFunction # 1024
embedding_fn = JinaEmbeddingFunction(
    model_name="jina-embeddings-v3",
    api_key="jina_4129a0d4fdd9469785d8a9728c6f4d9fUGPF0NemmXI_uVRHvnfGLImuEoyq"
)
# Cohere work 贾元春 贾迎春 贾宝玉 袭人
# from pymilvus.model.dense import CohereEmbeddingFunction # 1024
# embedding_fn = CohereEmbeddingFunction(
#     model_name="embed-multilingual-v3.0",
#     api_key="eoIhZt6kPWHtyfkdvsY7S14JEqIV1igjn8ymJDoX",
#     input_type="search_document",
#     embedding_types=["float"]
# )
# Voyage work voyage-multilingual-2(1024) 贾元春 贾迎春 贾宝玉 秦可卿 | voyage-2(1024) 贾元春 贾迎春 李纨 贾探春 | 
# voyage-3(1024) 贾元春 贾探春 贾宝玉 薛宝钗 | voyage-large-2(1536) 元 宝 迎 惜  | voyage-lite-02-instruct(1024) 元 黛 惜 探
# from pymilvus.model.dense import VoyageEmbeddingFunction # 1024
# embedding_fn = VoyageEmbeddingFunction(
#     model_name="voyage-multilingual-2", # https://docs.voyageai.com/docs/embeddings
#     api_key="pa-ReOQxAJwGywtO4bfpQVnjyJv5uHsqnBTC0ym8DE73Yg"
# )
# bge-m3 not work on some machine
# from pymilvus.model.hybrid import BGEM3EmbeddingFunction # model size 8G
# embedding_fn = BGEM3EmbeddingFunction(
#     model_name='BAAI/bge-m3', # Specify the model name
#     device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0', on cpu speed is very slow
#     use_fp16=True # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
# )
# OpenAI no key
# from pymilvus.model.dense import OpenAIEmbeddingFunction
# embedding_fn = OpenAIEmbeddingFunction(
#     model_name: str = "text-embedding-ada-002", 
#     api_key: Optional[str] = None,
#     base_url: Optional[str] = None,
#     dimensions: Optional[int] = None,
# )

DIM = 1024
docs = [
        # "Artificial intelligence was founded as an academic discipline in 1956.",
        # "Alan Turing was the first person to conduct substantial research in AI.",
        # "Born in Maida Vale, London, Turing was raised in southern England.",

        # 'This is a document about pineapple',
        # 'This is a document about oranges',

        # "（精读本）225没有想到，第四十回刘姥姥的一句话，竟引来贾母要惜春画下大观园。更没有想到，刘姥姥第四十回已经是二进贾府，其后再没来过贾府。一个老村妇的话，贾母竟然在第五十回三次催促贾惜春赶快画画。姥姥来不来还不知道呢？贾母为何就如此上心？宁愿为难孙女，也要把这画画出来呢？还是从头说起。第四十回，刘姥姥是这样说的：刘姥姥念佛说道：“我们乡下人到了年下，都上城来买画儿贴。时常闲了，大家都说，怎么得也到画儿上去逛逛。想着那个画儿也不过是假的，那里有这个真地方呢。谁知我今儿进这园里一瞧，竟比那画儿还强十倍。怎么得有人也照着这个园子画一张，我带了家去，给他们见见，死了也得好处。”第一，刘姥姥夸大观园之美，深得贾母之心。第二，刘姥姥为大观园留念想的办法，触动了贾母。大观园为谁而建？为元妃省亲而建。贾元春最思念的是谁？是家人，是家，是大观园。因此，刘姥姥这话，贾母第一个想起的就是她那地位尊贵，却不得见人的大孙女。因此，她立马这样表态：贾母听说，便指着惜春笑道：“你瞧我这个小孙女儿，他就会画。等明儿叫他画一张如何？”贾母这话，应景的成分很大。第一，惜春不可能一天就画好。第二，贾母也并没有说画好了是要送给刘姥姥，那个时代也没有复印机。所以，可以这么说，刘姥姥的话触动贾母，决定要惜春画大观园，可是这画不是为了刘姥姥，而是为了贾元春。贾母决定让惜春画一幅大观园图景，画上大观园的楼台山水，画上众姐妹，进献给贾元春，聊解相思之苦。只有这样的事情，才会让贾母如此上心，直到第五十回，还又问：“你四妹妹那里暖和，我们到那里瞧瞧他的画儿，赶年可有了。”众人笑道：“那里能年下就有了？只怕明年端阳有了。”贾母道：“这还了得！他竟比盖这园子还费工夫了。这是第一次催问。贾母为什么一定要年下要？就是因为过大年的时候，可以作为一份厚礼进献给贵妃。贾母明白，这才是贾元春最想要的礼物。所以，贾母如此上心，此时表达的却是对大孙女的一片深爱之情。所以，第二次催问：大家进入房中，贾母并不归坐，只问画在那里。惜春因笑回：“天气寒冷了，胶性皆凝涩不润，画了恐不好看，故此收起来。”贾母笑道：“我年下就要的。你别托懒儿，快拿出来给我快画。”这里依然是不松口，要惜春画出来。可是，就在这时，发生了一件事情，就是贾母夸赞薛宝琴长得好看，又问生辰八字。薛姨妈以为贾母是看上了宝琴要给宝玉做媳妇。这个我分析过，不是的。贾母其实是要给老世亲甄家的那个宝玉做媒。薛姨妈误解了贾母的意思，说出宝琴已经定亲，王熙凤打岔，这个话题貌似就过去了。所谓：大家又闲话了一会方散。一宿无话。然而，这是大障眼法。这一宿不仅不是无话，而是话很多。因为第二天，贾母第三次催画，催画的调子却变了：次日雪晴。饭后，贾母又亲嘱惜春：“不管冷暖，你只画去，赶到年下，十分不能便罢了。第一要紧把昨日琴儿和丫头梅花，照模照样，一笔别错，快快添上。”第一，贾母依然年下要画。第二，但是画的内容，什么大观园啊，什么众姐妹呀，都不重要了，重要是要画上宝琴还有梅花。这是为什么？不妨先来问问，薛蝌带妹妹到京完婚，怎么连亲家梅翰林家外任不在京城都不知道？闹这样的笑话？既然梅家不在京城，为什么不立马赶去梅家放外任的地方？薛蝌这样，到底是像带妹妹来完婚还是像来投靠婶娘薛姨妈进而投靠贾府？居然住下就不走了？其实真实的情况应该是，自从薛家二公死后，薛家败落，梅翰林已经有悔婚之意。薛蝌急了，才打着送妹完婚的名义上京来投靠贾府，寻求解决办法。刚好，贾母夸宝琴，薛姨妈说破宝琴婚事，于是机会来了。众人散去，我觉得应该是王夫人陪着薛姨妈去给贾母汇报了的，说明了宝琴的婚姻危机。因此，贾母灵机一动，原本要惜春画大观园给元妃解相思之苦的初衷变了，变成画一幅宝琴的肖像，借过年去宫内觐见元妃的时机，给元妃看这幅宝琴的画，借机（梅花）提起宝琴和梅家结亲的事，委婉的请求元春出手相助。 果然，到第五十七回，宝钗就这样对邢岫烟说了：“偏梅家又合家在任上，后年才进来。若是在这里，琴儿过去了，好再商议你这事。离了这里就完了。如今不先定了他妹妹的事，也断不敢先娶亲的。如今倒是一件难事。“此时借助贵妃的威势，外任的梅翰林已经表态了，后年回京便和薛家完婚，宝琴的婚姻危机，就这样化解于无形。 这便是贾母催画的玄机。一宿无话，其实蕴藏着太多太多的深意。",
        # "（精读本）298 第六十二回，宝玉的生日，引出宝琴的生日，又引出平儿的生日，再引出邢岫烟的生日。原来，这四人竟是同一天生日。  巧啊。你说曹雪芹为什么要这样写呢？  这还不算，还有呢。探春说： “倒有些意思，一年十二个月，月月有几个生日。人多了，便这等巧，也有三个一日、两个一日的。大年初一日也不白过，大姐姐占了去。怨不得他福大，生日比别人就占先。又是太祖太爷的生日。过了灯节，就是老太太和宝姐姐，他们娘儿两个遇的巧。三月初一日是太太，初九日是琏二哥哥。二月没人。” 这里面，又牵带出几个人的生日。元春是大年初一，竟然和她的曾祖第一代荣国公是一天。贾母和宝钗是正月十六。三月初九是贾琏生日。 然后，探春一句二月没人，又引出黛玉和袭人的生日。所谓： 袭人道：“二月十二是林姑娘，怎么没人？就只不是咱家的人。”探春笑道：“我这个记性是怎么了！”宝玉笑指袭人道：“他和林妹妹是一日，所以他记的。”  原来二月十二是黛玉和袭人的生日。 这些人的生日，如此集中，如此巧合，到底有什么意义？ 我以为，答案在第二回贾雨村所说大仁、大恶和正邪两赋三种人中正邪两赋之人。所谓： “今当运隆祚永之朝，太平无为之世，清明灵秀之气所秉者，上至朝廷，下及草野，比比皆是。所余之秀气，漫无所归，遂为甘露，为和风，洽然溉及四海。彼残忍乖僻之邪气，不能荡溢于光天化日之中，遂凝结充塞于深沟大壑之内，偶因风荡，或被云催，略有摇动感发之意，一丝半缕误而泄出者，偶值灵秀之气适过，正不容邪，邪复妒正，两不相下，亦如风水雷电，地中既遇，既不能消，又不能让，必至搏击掀发后始尽。故其气亦必赋人，发泄一尽始散。使男女偶秉此气而生者，在上则不能成仁人君子，下亦不能为大凶大恶。置之于万万人中，其聪俊灵秀之气，则在万万人之上，其 云 邪谬不近人情之态，又在万万人之下。若生于公侯富贵之家，则为情痴情种，若生于诗书清贫之族，则为逸士高人，纵再偶生于薄祚寒门，断不能为走卒健仆，甘遭庸人驱制驾驭，必为奇优名倡。” 上述之人，一则年岁如此相近，生日如此苟同，说明一拨一拨皆是秉正邪二气所生之人。  二则，又有太祖贾源，贾琏，宝玉三位男性，贾源乃是荣府始祖，贾琏和宝玉是四世孙，一个滥淫，一个意淫，一邪一正，推及贾赦贾政，莫不如此，归于贾源。  三则元春之于贾源，喻荣府之元气，周而复始，归于女子。  宝钗之于贾母，袭人之于黛玉，则喻本为一源却不能相容，所谓“正不容邪，邪复妒正，两不相下，亦如风水雷电，地中既遇，既不能消，又不能让，必至搏击掀发后始尽”也。  而宝玉宝琴平儿岫烟，又为一气，是谓“所余之秀气，漫无所归，遂为甘露，为和风，洽然溉及四海”也。  自然还有他人，雪芹只是略点染一笔，提纲挈领，说明要义而已。非要说全说白，反而无趣。  雪芹要说的，是他所写，非为大仁大恶者立传，而通篇都是正邪禀赋之人，是情痴情种，是逸士高人，是奇优名倡。  叹此番心意，终为陈寅恪先生之《柳如是别传》所得。",
        # "（精读本）305 读《红楼梦》，我们不仅要看到贾府的堕落，也要看到贾府的好处。应该说，贾府起家是兄弟二人，却能够历经五世不散，是有其独到之处的。这个独到之处，就是孝悌。按照小说里的话说，就是儿子怕老子，弟弟怕哥哥。这是一种秩序。 所以贾母斥责贾赦老不正经，贾赦是含羞且愧。贾母说一生没养个好儿子，贾政就说自己无立足之地。贾珍贵为族长，在贾母贾赦贾政面前，也得恭恭敬敬。贾珍贾蓉和贾赦贾政贾琏宝玉贾兰是隔了三四层的兄弟，却表现得异常亲密。 如果还要给这种精神扩大一点内涵的话，那就是善待女儿和下人。 无论是贾敏还是四春，甚至黛玉湘云宝钗，贾府是竭尽所能善待的。金钏儿投井，贾政闻之惊悸，有一番反躬自省。贾府的下人婆子丫鬟，在贾府得到的是好于一般家庭的待遇。 这就铸就了贾府有德之家的声望，积四世，方出了一个贤德妃。从这个角度看，既是偶然，也属必然。 因为贾府的这种精神，是高度契合于当时的皇帝所提倡的以孝治国的。 无论是老太妃之死，还是准许省亲，又或者每月祖母母亲可入宫探望一次，无不体现这种孝悌精神。 第六十四回，有一个细节，就深刻表现了这种孝悌精神。贾琏随贾母王夫人等去给贾敬送殡回来，宝玉听闻，是这样的：  只见有人回道：“琏二爷回来了。适才外间传说，往东府里去了好一会了，想必就回来的。”宝玉听了，连忙起身，迎至大门以内等待。恰好贾琏自外下马进来。于是宝玉先迎着贾琏跪下，口中给贾母王夫人等请了安，又给贾琏请了安。二人携手走了进来。只见李纨、凤姐、宝钗、黛玉、迎、探、惜等早在中堂等候，一一相见已毕。  此时贾琏回来，代表是贾母王夫人。所以宝玉的礼节，纹丝不乱。这里表现的父母兄弟之间的孝悌精神，也是令人动容的。弟弟给哥哥跪下，请祖母母亲的安，又给哥哥请安，然后兄弟携手进入。平辈的媳妇姊妹，早已在内堂迎候，好一派大家族其乐融融的景象。  宝玉是贾府尊贵之人，却始终懂礼节不自大，这便是宝玉可爱之处。记得贾母和甄家婆子也说，她疼宝玉，不惟是长相可爱，其实也是他懂礼节知孝顺。如果一味胡搅蛮缠，早也就淡了。此处很好的阐释了这一点。  只可惜，宁荣二公以武功起家，以孝悌治家，成于有德，也败于有德。贾赦贾珍贾琏贾蓉表面一套，背后一套，毁了贾府积五世的功德。令人可叹",

        # "林黛玉的嫁妆，究竟藏在了哪里？刘姥姥一句浑话揭开了谜底！ 在《红楼梦》中，林黛玉的嫁妆一直是一个令人猜测的谜题。这些嫁妆究竟被存放在了何处？在小说中，虽然没有明确交代，但通过小说中的种种线索和情节，我们可以揭开这个谜底。刘姥姥的一句话更是直接暗示了嫁妆的存放地点。 林黛玉的悲惨命运 林黛玉是在红楼梦中，是一位极具有才的女性角色。她美丽而聪慧，深受贾母的宠爱。林黛玉是林如海的独生女儿，因此她的嫁妆无疑是一项重要的财产。嫁妆在古代社会被视为女子嫁入新家的重要资产，因此它的去向成为了小说情节中的一个关键因素。林如海的去世对林黛玉，以及整个贾府来说是一场巨大的变故。在林如海重病之际，贾母派人将林黛玉送回了林家，顺便让贾琏跟随前去，但是贾母为什么会让贾琏也跟随前去呢？ 简单来说就是，需要确保林家的遗产得到妥善安排。这也是贾母为什么派贾琏前去的一个原因。贾母深爱林黛玉，此时的林黛玉已经失去了父母，贾母同时又是林如海的岳母，有着共同的立场，她肯定也希望林黛玉的家产能够妥善安置。 林黛玉的嫁妆藏在了哪里 随着林如海的去世，贾府的成员都明白，林黛玉将长期留在这里。林黛玉的嫁妆必然也会随着她一同进入贾府。但由于贾府的经济压力和贾府成员之间的权谋，林黛玉的嫁妆可能会引起争夺。特别是当贾府开始兴建大观园来迎接元春省亲的时候，就曾想过动用林黛玉的嫁妆。 但是贾政是一个正直的人，而且与林如海有着深厚的友情，因此不会同意使用林黛玉的嫁妆来填补自家的亏空。贾母也是真心疼爱林黛玉的，不愿意让其他人觊觎她的嫁妆。而王夫人和王熙凤可能会打这种主意，但出于面子和尊重，暂时不会采取行动。林黛玉进入贾府后一直住在贾母的房间，综合考虑所有情况，贾琏带回的嫁妆可能由贾母亲自保管。 贾母肯定会非常小心地保护这些嫁妆，因为她深爱林黛玉，不希望让其他人有机会觊觎她的财产。古代女子的嫁妆通常包括许多贵重物品，因为林黛玉是林如海的独生女儿，所以她的嫁妆必然不会少。那么，林黛玉的嫁妆的储存地点究竟在哪里呢？虽然小说中没有明确交代，但通过一些细节和线索，我们或许可以找到答案。 在小说中，刘姥姥的一次探访，也许能为我们揭示林黛玉的嫁妆藏在了何处。刘姥姥是一个善于观察的老妇人，她在贾府里游览时，不禁感叹房间里的富丽堂皇。她特别提到了贾母的房间，说：“那柜子比我们那一间房子还大还高” ，她还谈到了房间里的梯子，猜测它可能是用于开顶柜，以存放东西。 这段描写提供了一个关键的线索，贾母的房间之所以如此富丽堂皇，柜子如此巨大，是因为它不仅仅是用来存放贾母的个人物品，还可能包括了林黛玉的嫁妆。刘姥姥的提到的大柜子，或许就是黛玉嫁妆的藏身之地。这种假设合理，因为在古代，家庭中的贵重物品通常是放在柜子中，而顶柜则是更加隐蔽和安全的地方。 贾母对林黛玉的特殊爱护，也可以解释她对嫁妆的特殊保管。贾母希望将这些财产妥善保存，以保障林黛玉的未来。同时她也提醒着其他人，不要打林黛玉嫁妆的主意。 写到最后 虽然小说中没有明确交代嫁妆的具体存放地点，但通过综合分析小说中的情节和线索，我们可以合理猜测，林黛玉的嫁妆很可能藏在贾母的房间中的顶柜内。这种安排既体现了贾母对林黛玉的爱护，也为后续情节提供了重要的支持。",
        # "林黛玉的家产被贾家侵占挪用？看看黛玉自己是怎么说的 有一个重要的情节，曹雪芹在书中没有明确交待，就是林黛玉的父亲林如海去世以后，林家的家产究竟去了何方？多数红楼梦爱好者认为被贾家给侵占挪用了，黛玉给父亲奔丧的时间又恰好和修建大观园的时间相契合，有人也举出了很多的事例，主要就是贾琏说过一句，这个时候再发个二三百万两银子的财就好了，这里不多赘述。我们今天主要来看黛玉在贾府的吃穿用度是个什么水平，她真的很可怜吗？邢岫烟家境贫困经常靠当衣服贴补日常用度，显然黛玉不是，有人说贾母疼黛玉，黛玉的日常用度自然是低不了的，贾母到底有多疼黛玉？事实竟是毫不避嫌的直接送钱给黛玉。书中写佳惠对小红笑道：“我好造化！才刚在院子里洗东西，宝玉叫往林姑娘那里送茶叶，花大姐姐交给我送去。可巧老太太那里给林姑娘送钱来，正分给他们的丫头们呢。见我去了，林姑娘就抓了两把给我，也不知多少。你替我收着。”便把手帕子打开，把钱倒了出来，红玉替他一五一十的数了收起。可见黛玉给佳惠的钱真的是不少，黛玉并不能像现代人似得，拿着钱可以到处逛，随意的上街买东西更是不行，贾母给黛玉钱是不是显得有点俗套呢？书中说的很清楚，黛玉并不是不会管家，更不是对钱没有概念。书中有一次就提到黛玉跟宝玉说，我也暗地里给你们家盘算过，竟是出的多进的少，宝玉还宽慰说，“少了谁的也少不了我们俩的”。黛玉即会给贾府计算日常用度，自己父亲遗产的去向她又怎会不知道、书中还有一回提到黛玉和宝钗在聊燕窝的话题，黛玉这样对宝钗说——“你方才叫我吃燕窝粥的话，虽然燕窝易得，请大夫，熬药，人参肉桂，已经闹了个天翻地覆，这会子又兴出个什么熬燕窝粥来，老太太，太太，凤姐三个人便没话说，这些底下的婆子丫头们，未免嫌我太多事了。”黛玉的这番话的意思是吃燕窝没有问题，完全吃得起，只是去请大夫，找人熬药需要用到下人，因为自己不是贾府的正经主子，这些婆子丫头总是抱怨自己多事，所以才一切从简的。这里黛玉还搬出来三个人，贾母，王夫人，还有王熙凤，说这三个人没话说，是什么意思？难道是因为这三个人最疼黛玉？还是因为这三个人在贾府当家，掌管金钱呢？显然后者更有说服力，当初黛玉和贾琏办完父亲的后事之后，是一同回的贾府，黛玉自然知道是带了钱过来的，贾母知道黛玉的钱最终的去处，或许大部分都拿去修建了大观园，贾母及贾家众人对黛玉在钱财上觉得有亏欠，所以经常给黛玉送钱，那王夫人和王熙凤就不用说了，各位读者都知道大事情王熙凤做不了主，都是在王夫人的授意之下，更兼贾琏还是亲自接手之人，具体贪了多少也只有她夫妻二人自己知道了。贾府对林家这笔钱财的使用和去处，黛玉心知肚明，打小又无父无母，个人的婚事又无人做主。所以黛玉才会发出“一年三百六十日，风刀霜剑严相逼”的千古绝唱。",
        # "父母留下百万家产，为何林黛玉名下只有一份嫁妆，钱都去哪儿了？	林如海可能有百万家产，但哪怕有荣国府撑腰，贾琏也不可能带走其全部财产。这是为什么呢？林如海应有嗣子，黛玉作为独女，既非父母遗产的唯一，也非第一继承人，最终名下也仅仅得了一份嫁妆而已。一、林如海应有嗣子，承担延续香火之责。有人会说书中讲林如海病重之际，贾琏和黛玉就陪在其身边，他怎么会不顾骨肉至亲的女儿，胳膊肘往外拐，将家产拱手分给外人呢？如此林如海生前，在遗嘱中安排立嗣的可能性，真的就可以忽略不计吗？其实不然。图片林如海如非突然死亡，林如海生前应该安排了立嗣之事。古代男尊女卑的封建男权社会，在传统男婚女嫁的婚姻制度下，女子＂出嫁从夫＂，唯有儿子才能履行传宗接代的任务，承担祭祀之责。《围炉夜话》中有句：＂古语道，有子万事足。＂生十个女儿，有十个女婿也当不了一个儿子，正所谓＂家有万金不富，五个儿子是绝户＂。黛玉是独女，林家为绝户，必须由嗣子传宗接代，承担祭祀之责。所谓不孝有三，无后为大，死后断了香火万万不能。毕竟古人视死如生，女儿终归外姓人，外孙和女婿不会主持祭祀事宜，祖宗香火就断了。图片武则天和狄仁杰正如历史上女皇武则天，因狄仁杰一句：关于后世祭祀之事，从古至今绝无子侄祭姑姑一说，就立刻断了传位于武家子弟的想法。宝玉最爱林妹妹，也孝字为先将她排在老祖宗和父母之后，饱学的林如海就算再疼女儿，也不会忘了祖宗。所以，林如海托孤于贾府之外，必然会安排过继儿子的事，财产也会做出分配。绝不可能说将全部财产归于将要远嫁的黛玉一人。图片林黛玉何况，林如海生前未立嗣子成立，不等于他死后也没有。首先，林黛玉是未出阁的独女，按旧时礼俗，林如海死后必须马上要从族中子弟中立一嗣子，主持丧葬事宜。于情于理，立嗣都是贾琏和贾府反对无效，并且必须表示认可的事情。其次，有产有业的林家成了＂绝户＂，族人中觊觎者自然会站出来争夺。贾敏早亡已故，但有几个姬妾姨娘还在，完全可以由其中一位过继族中子侄为子，只要拥有了嗣子身份，就可以合法地继承林家产业。这对姨娘来说相当于丈夫死后扶正，由身份卑微的妾室，有了成为当家主母的机会，当然求之不得。至于被过继的子侄白得一份家产，只需多供几位林氏祖宗，根本算不得问题。对于林氏宗亲来说，终归肥水不流外人田，也是好事。图片《红楼梦》剧照所以哪怕林如海夫妇已去世，仍然可由族长和黛玉的姨娘及其娘家人，加上有意出继儿子的一家，联手完成立嗣之举，取得分家析产的合法资格。另外，贾府不可能瞒天过海，阻止林家人立嗣。林如海这样有高级官员的人死亡，必然大张旗鼓地办丧事，瞒是瞒不住的。林如海能娶上荣国府的千金贾敏，自然门当户对，家族实力不容小觑，不比草根出身的探花郎。林家近支没有了，旁枝还有不少。封建社会大家族的族权有很大的权利，即便姨奶奶势弱不提，眼看着林家香火断绝这样的事，家族长辈绝对不能容忍的。图片《红楼梦》剧照林氏同宗流寓不定，但参见贾家在金陵的情况，苏州一定有人看老宅。既然远在京城的贾琏能到，下葬需要四十九天，一个多月，林氏宗亲得知消息也晚一点赶到。林氏同宗流寓不定，但是苏州一定有人看老宅，参见贾家在金陵的情况，既然远在京城的贾琏能到，下葬需要四十九天，一个多月，林氏宗亲得知消息也晚一点赶到。二、林如海的嗣子，比黛玉更有优先继承权。如《清会典事例.户口.旗人抚养嗣子》也规定，嗣子有继承宗祧、继承遗产的权利。也就是说，嗣子是过继的宗亲之子，比起死者的亲生女儿更有优先的财产继承权。就作者曹雪芹来家来说，在其曾祖母李氏的儿子曹寅和孙子曹颙死后，就过继了兄弟曹荃的儿子曹頫，接任江宁织造一职。而《大明令.户令》中关于财产继承规定：死者没有儿子，可由其未亡人以及宗族来为他设定一个嗣子，一般从死者同宗的子侄中选出。如果没有为死者立嗣的话，全部遗产则由女儿均分。也就是说，只有在没有为林如海立嗣的情况下，孤女黛玉才能继承父母所有财产。但如前所说，林如海死后一定要过继嗣子延续香火，继承其家产。图片《红楼梦》剧照三、林黛玉继承父母遗产的过程，实际上是贾林家族势力的较量过程。俗话说强龙斗不过地头蛇，荣国府虽是豪门，但林氏宗亲更占地利人和，又非一般小人物，不可能让贾家人一手遮天。苏州的地方官表示很为难：贾家皇亲国戚朝廷有人，林家世家大族，以后低头不见抬头见，两方都得罪不起。因此，作为黛玉监护人的贾府，在朝中及江南的势力很大，不得不做出让步。允许林氏宗族以嗣子方的身份，参与林如海的遗产分配。比如供奉祖宗牌坊的祖产大宅房屋肯定不行的，地产房产坟田一类肯定不能全部拍卖带走的。荣国府长孙贾琏以贾敏侄儿，同知官家身份去操办林如海的后事，确定给黛玉的嫁资外，可替未出阁女儿林黛玉嫁妆的理由，争得属于贾敏的金银细软或古董分红等，以及借权势处理掉一些田产，占不少便宜。贾琏当然不会放过存款、应收账款和收藏等金银细软易变现的资产。图片贾琏理由很简单，可言是贾敏的陪嫁之物，或伪称贾家存银，或借款，从而以追索欠款为由，变卖林家名下的田产房屋，靠强势夺产。比如贾蔷去采办小戏子时，就提取的是存放在江南甄家的十万两银子。贾琏在荣国府理事，经验丰富办事妥当，贾母派他去也放心。期间贾琏派小厮回府取大毛衣服，不过是个由头，主要是听贾母关于林家财产处置的训示。四、父母纵有百万遗产，未出阁的林黛玉只可能分得一份嫁妆。贾府侵占林如海的财产，很大程度上并不是以黛玉继承父母遗产的形式进行的，除了贾敏的私人物品之外，更多是以追索贾府欠款等理由下开展的。当林妹妹回到贾府后，大部分的财产都成了贾琏口中的横财被侵占，真正明确属于黛玉名下的财产，也只有一份嫁妆罢了。即便如此，细论起来贾家也不理亏。至少贾琏给林黛玉争取到了一笔丰厚的嫁妆。黛玉的嫁妆被带回贾府，交由外婆史太君保管，人也受其精心庇佑，生活有了保障。图片林黛玉古代在民间称无子户为＂绝户＂，一般＂绝户＂留下的遗产，很难归在女儿或遗属的名下，吞孤噬寡的现象很普遍。倘若不是荣国府这样豪门娘家人的身份出场，或是没落子弟无权无势，说不上话的娘家人，林氏宗亲勾结官府霸占财产，象征性地给黛玉一点钱就打发掉了。贾家无人的话，黛玉这笔大幅度缩水的嫁妆钱，甚至会让其姨奶奶或族人保管，待出阁不知又要消耗多少去了。所以年幼失怙的富家女，如果没有亲人撑腰，结局也颇为凄凉的。",
        # "《刘心武妙品红楼梦》| 林黛玉的巨额家产，被谁侵占了？	昨天，我们读到的是贾宝玉的玉石之谜与复杂人格。曹雪芹塑造这样一个角色，使得我们相信，在那个时空，有那样一个生命度过他的人生。这个秉正邪二气而生的人，其人格因素中，既有圣洁的形而上，也有粗鄙的形而下。林黛玉是否有继承家产？她是在得知贾宝玉和薛宝钗成亲的情况下悲愤而死，还是选择其它某种方式离开人间？接下来，让我们开始今天的共读吧。她的家产被谁侵占了？作为《红楼梦》女一号，关于林黛玉的文字背后，隐藏着大量鲜为人知且难以解释的谜团。第十二回，写到林黛玉的父亲林如海病重，她在贾琏的护送下回扬州探望。没多久，林父不治身亡。他有没有留巨额遗产？如果有，林黛玉能否继承这笔遗产？这个问题很重要，因为关系到林黛玉的经济状况，也就是家业根基。林如海祖上三代都由皇帝封了贵族头衔，他自己科举出身，高中探花，后来在扬州当巡盐御史。这是一个肥缺，专门管理盐的开采、配置、运送及税收。按照常理，林如海死后一定留有大笔遗产。在那个时代，女儿以嫁妆的形式分割家庭财产。林黛玉还没结婚，当然有资格继承遗产。而且，林黛玉的母亲贾敏去世后，林如海没有续弦，作为林家的独生女，林黛玉不但拥有遗产继承权，而且没有人跟她争夺。可是仔细读《红楼梦》就会发现，林黛玉一点银子都没得到，以至于后来在贾府寄人篱下，无依无靠。从经济地位来说，她成为贾府里面最悲苦的一个小姐。贾府的几个小姐都有父母作为靠山，家中也有财产。薛宝钗后来住进荣国府，靠的完全是亲戚情分，薛家经济上很强大，城里有房子，随时可以搬过去住。成为孤儿的林黛玉，没有任何经济外援，住进贾府之后，“一无所有，吃穿用度，一草一纸，皆是和他们家姑娘一样”。而这，全凭贾母对她的格外疼爱，经济上为她保驾护航。应该由林黛玉继承的遗产，究竟去了哪里呢？林如海病逝后，是贾琏带着林黛玉将灵柩护送回原籍苏州的，书上写道：“诸事停妥，贾琏方进京的。”所谓“诸事停妥”，包括贾琏以监护人身份争取到了遗产这件事。按理，他应该将这笔遗产全部折合成银子，交给荣国府总账房保存，等林黛玉出嫁时取出。从苏州回来后，贾琏很快操办元春省亲一事，不过他并未将带回的林氏遗产挪用于建造大观园，而是自己侵吞了，因为他是一个“油锅里的钱还要找回来花”的贪婪之人。《红楼梦》第七十二回，贾琏和王熙凤聊到钱财话题时，贾琏辩不过王熙凤，以这样一句话收场：“这会子再发个三二百万财就好了。”“这会子”，是相对于“那会子”而言的，“那会子”是哪会子？应该就是他陪林黛玉去扬州那阵子。林如海病逝后，贾琏替林黛玉争取到应得的财产后，有可能形式上往官中交了一点，其他就和王熙凤私吞了。林黛玉，贾母血缘上最亲近的人我们读《红楼梦》时，也许会产生一个疑惑：贾宝玉和林黛玉是姑表兄妹，血缘如此接近的两个人，相爱本就是一种避忌。贾母为何不阻止，还主张他俩结婚呢？刘心武研究《红楼梦》有一个基本看法，就是这部小说带有自传性、自叙性，许多人物都有生活原型，“真事隐”“假语存”是它的艺术宗旨。宝玉和黛玉虽然被设定为姑表兄，但在现实生活中，这两个角色的原型也是这样一层关系吗？要搞清这个问题，首先必须了解贾母这个人物的原型。贾母的原型，是康熙苏州织造李煦的妹妹，嫁给江宁织造曹寅。正当她享受富贵生活时，曹寅因患疟疾去世。曹寅去世后，江宁织造这个官职由他和李氏所生的唯一一个儿子曹顒继承。可是天公不作美，没过几年，曹顒也去世了。康熙跟曹寅感情深厚，对未亡人李氏也关怀备至，于是让李煦去安排，从曹寅的侄子当中选一个过继给李氏，接着当江宁织造。李煦推荐了曹寅的侄子曹頫。这个曹頫，与李氏没有直接的血缘关系，转化到小说里面的艺术形象，却成了贾母的亲儿子贾政。从书中猜灯谜时贾政承欢膝下、贾政痛打贾宝玉等情节不难看出，贾母与贾政这对母子之间，只有礼节，没有感情。贾母的另外一个儿子贾赦，在现实生活中是贾政原型的亲哥哥，并没有与之一起过继到李氏门下。曹雪芹在写作当中遵循一个原则，当小说里的人物设计和生活实际难以协调时，他会选择牺牲小说本身逻辑的圆满，去照顾生活的真实。换言之，《红楼梦》中贾母的亲儿子贾政，其生活原型是贾母原型李氏过继的儿子。那么，贾宝玉的原型，也就并非贾母原型的亲孙子。由此可见，在现实生活当中，贾宝玉和林黛玉这两个人物原型的血缘其实离得比较远。生活当中的李氏有一个亲女儿，在小说里面转化为贾敏，生了一个女儿，这个女儿转化到小说里被设计为林黛玉。所以说，在众多儿孙当中，林黛玉才是和贾母血缘最近的人。林黛玉初进荣国府，贾母见到这个外孙女时激动异常，“一把搂入怀中,‘心肝儿肉’叫着大哭起来。 ”王熙凤为了讨贾母欢心，夸林黛玉的气派不像老祖宗的外孙女，分明是嫡亲的孙女。因为她知道，在血缘上，林黛玉是跟贾母最近的一个生命。贾母之所以愿意让林黛玉嫁给贾宝玉，其中一个原因，就是她不存在血缘相近不宜结婚的心理障碍。沉湖仙遁关于林黛玉的结局，在通行本《红楼梦》八十回之后，贾宝玉中了调包计与薛宝钗成亲，林黛玉知道后焚稿断痴情，悲愤而死。刘心武则认为，“焚稿断痴情”堪称续书中最成功的部分，但并不符合曹雪芹的原笔原意。小说里面对宝玉和黛玉的身份有特殊设定，前者是天界赤瑕宫的神瑛侍者，后者原是天上的一棵绛珠仙草。绛珠仙草下凡成为林黛玉，是为了用一生的眼泪，偿还神瑛侍者的灌溉之恩。对于有着仙界身份的林黛玉，如何安排她的结局，一定是曹雪芹精心设计的内容。前八十回中最能体现她生活状态与精神气质的黛玉葬花，是深入分析曹雪芹创作意图的文本。除了葬花这一艺术行为，还可以从林黛玉日常生活的多个片段，比如为燕子留门、教鹦鹉念诗这些地方可以看出，她是诗意生存的一个人，她将生活充分地艺术化。既然她的生存是诗意的生存，一旦还泪之旅抵达终点，也一定会选择诗意的消逝方式离开这个世界。刘心武通过对《红楼梦》探佚，认为林黛玉之死，应该在贾母死亡之后。因为贾母死后，林黛玉失去靠山，“金玉姻缘”又在紧锣密鼓地筹备，加上贾府中的小人暗中实施的陷害，她的处境变得非常糟糕。最关键的是，她来人间的还泪目的已经完成，该回天界去了。而她采取的回归方式，是沉湖。不是跳湖自杀，而是沉湖仙遁。对此结局，前八十回好几个地方设下伏笔。比如第七十六回，中秋之夜，黛玉和湘云在湖边联诗，湘云说“寒塘渡鹤影”，黛玉对的是“冷月葬花魂”。早在第二十三回，林黛玉初进大观园住进潇湘馆，和贾宝玉偷读《西厢记》，听见远处学戏的小姑娘在唱曲，让她想起“花落水流红”、“水流花落两无情”、“流水落花春去也”这些诗句，其实就是作者对青春少女最后在水中消逝的伏笔。还有第三十七回，探春组织海棠社，每个人都要取别号，林黛玉的别号是“潇湘妃子”。潇湘妃子是舜的两个妃子，舜出去巡查，不幸死在外地，这两个妃子非常悲痛，将泪洒落在竹子上，最终“泪尽入水”。也就是说，她俩泪水哭干后，死于江湖。所以，潇湘妃子这个别号，也暗示了林黛玉沉湖而死。林黛玉是神仙下凡，她在人间所谓的死亡，实际上是复归天界。沉湖后，不会有尸体，只留下衣服和钗环，因为她是仙遁。结语 今天，我们读到的是林黛玉的家产、血缘和沉湖之谜。刘心武通过探佚《红楼梦》，认为原本属于林黛玉的家产被贾琏私吞，导致她在贾府成为最悲苦的小姐。林黛玉是贾母血缘上最亲近的人，因而得到贾母特殊的珍爱。还泪完成，她以沉湖仙遁的方式返回天界。",
        # # "红楼梦：把林黛玉送入贾府，是林如海夫妇和贾母的秘密约定	再一次读《红楼梦》，我心中有一个奇怪的疑问，那就是：林黛玉为什么要进贾府？按理来说，这本不应该是个问题，毕竟作者在文中已经有明确的交待。林黛玉的母亲贾敏去世，贾母因为心疼外孙女，才派人把林黛玉接入贾府。照这个说法，林黛玉到贾府，性质完全是客居、走亲戚。但奇怪的是，从后文的情节发展，以及诸多的细节描写来看，作者似乎又在有意无意的勾着我们怀疑这种说法。比如，最让我们感到奇怪的一个问题，按照书中的时间线，林黛玉进贾府的时候不过才六岁，等到林如海死的时候大约是十岁左右，四年多的时间里林黛玉为什么从来不回家探亲？1补充一点，关于黛玉的年龄问题，有很多证据可以说明，我这里只说一条。黛玉出场的时候是五岁，一年后母亲贾敏去世，黛玉此时是六岁，而黛玉正是在这一年进了贾府。与此同时，冷子兴在说起贾府的时候，提到此时的贾蓉是十六岁，也就是说贾蓉比黛玉大十岁。而对于贾蓉的年龄，秦可卿死的时候 ，作者明确告诉我们他是二十岁，也就是说此时距离黛玉进贾府已经过去了四年，所以黛玉应该是十岁。也就是说，黛玉此时在贾府已经待了四年。我们试想一下，黛玉到贾府如果是走亲戚，怎么会一住就是四年，而且中间从来都没有回家看望父亲？除此以外，疑点还有很多。黛玉进贾府的时候，母亲贾敏刚刚去世一个月，也就是说黛玉刚刚守完孝。林如海夫人新丧，心情肯定是不好，黛玉为什么偏偏在这个时候要离他而去呢？更奇怪的一点是，从林如海跟贾雨村的对话可知，在林黛玉进贾府以前，贾母早就派了船来接黛玉，只不过因为黛玉病未痊愈才没有走。这也就是说，在黛玉还在守孝期间，贾母就已经派了人来接她。这么着急着把人接走，这个事情就透着有些奇怪，这恐怕不单纯是贾母想外孙女那么简单了。而林黛玉临行之前，林如海对她说的一番话，也证明了黛玉此行，并不是走亲戚这么简单，而应该是托付。“汝父年将半百，再无续室之意；且汝多病，年又极小，上无亲母教养，下无姐妹兄弟扶持，今依傍外祖母及舅氏姐妹去，正好减我盼顾之忧，何反云不往？”林如海这番话说的入情入理，但是意思也很清楚，明明就是把林黛玉托付给了贾府抚养，所以用了“依傍”二字。甲戌本《红楼梦》第三回的标题，为“荣国府收养林黛玉”，这也明确的告诉了我们“收养”的性质。什么是“收养”？“收养”这个词，在古代的分量是很重的，为了让大家理解这两个字的含义，我在下面稍微费一点笔墨。比如，薛宝钗长期随母亲住在贾府，而一应生活开支都是自己负责，这种属于寄居，而不属于“收养”。像林黛玉因为母亲去世，贾母心疼外孙女，所以把她接到贾府长期生活，这种情况理论上应该叫做寄养，而不应该叫收养。这两者之间的区别在于，寄养的“抚养权”属于林如海，而贾府只是代替林如海尽抚养义务，而收养则说明林如海已经把“抚养权”交给了贾府。换句话说，只有林如海把女儿送给了贾府，才能称得上是“收养”。而从林黛玉在贾府的处境来看，与姐妹们同吃同住，拿相同的月例银子，这似乎也更符合“收养”的情况。这就更让人疑惑了，林如海只有这么一个独生女儿，而且林家又是公侯显贵之家，又不是说养不起女儿，何至于把林黛玉送给贾府呢？2有研究者推测，林如海可能是预感到了政治上的危机，所以才要把女儿远远地送走，目的是为了保护林黛玉不受到牵连。这种说法显然是没有道理的，先不说小说中没有任何证据，古代有严格的户籍制度，即便是林如海面临着抄家的风险，所以才把林黛玉送到了贾府，也不可能保住林黛玉不受牵连。毕竟，林黛玉入贾府是光明正大的，而且有贾雨村这个知情人亲自护送，贾雨村可不像是个能保守秘密的人。既然如此，我们应该怎样理解林如海的行为呢？这应该先从林如海这个人说起。林如海，姑苏人氏，祖上是列侯出身，他本身则是科甲正途出身，是前科探花。林如海曾担任兰台寺大夫一职，而兰台寺是御史台的早期称呼，明清时期改称都察院，兰台寺大夫可能是都察院御史大夫的变称。在林如海正式出场时，已经改任巡盐御史，而巡盐御史又是个极有油水的肥缺。由此可知，林如海官运亨通、财运亨通，非要说有什么缺憾的话，那就是没有儿子。“今如海年已四十，只有一个三岁之子，偏又于去岁死了。虽有几房姬妾，奈他命中无子，亦无可如何之事。”不孝有三，无后为大。“命中无子”的林如海，如今已经四十多岁，而且估计身体也不好，所以他亟待解决的应该是传嗣的问题，这在古代是任何一个男人都无法规避的事情。他又说自己没有续室之意，可见对生儿子已经不指望了。按照常理来说，这种情况之下，他应该有两种选择的可能性。第一种就是过继或领养，从族中其他堂兄弟的儿子中过继一个作为嗣子，以林如海的家世地位，相信这一点应该不难办到。第二种情况就是招赘，为独生女林黛玉招一个上门女婿，以后生的孩子直接姓林，继承林家的香火。从林如海的行为来看，他选择的应该是第二种，所以他把林黛玉“假充养子”，又在五岁的时候就请先生教她读“四书”。这分明就是想让林黛玉诗书传家，为林家培养下一代，毕竟只有参加科举的人才需要读“四书”。可既然想要招赘，怎么又把林黛玉送给了贾府，难不成是要在贾府招一个上门女婿？这显然是说不过去的，如果招上门女婿，应该让男方到林家才是。可是在这种背景之下，既然林如海选择了把林黛玉送到贾府，就说明这样做可以解决林家的传嗣问题。如果这样的话， 那就只剩下了一种情况，也就是宗法制度中的另一种传嗣方式——兼祧。而兼祧，正是曹雪芹所生活的乾隆年间制定的一项制度。3兼祧，也就是一人祧两门，一个人继承两个家族的香火。放在林黛玉的身上，也就是说，林黛玉嫁给贾宝玉，生下的孩子同时继承贾家和林家的香火。当然，还可以有另外一种情况，即林黛玉和贾宝玉生下了两个男孩，其中一个给贾家传嗣，另一个给林家传嗣。相比前两种可能而言，这种情况可能是更好的选择。因为林黛玉的嫡母贾敏已死，而从林如海四十一岁就决定不续弦来看，他的身体状况应该是很差。试想一下，林黛玉此时才六岁，不管林如海选择过继还是招赘一个外人进林家，以他的身体状况来看，万一他哪一天去世了，林家的所有家产就都落到了外人手中，而林黛玉也就变成了任人宰割的孤魂野鬼。像这样的事情，林如海不可能想象不到，所以把林黛玉嫁入贾家才是最优解。当然，这个决定应该在贾敏活着的时候就定下了，只有她跟贾母沟通这样的事情才是合理的。而且对于贾敏来说，这是一举三得的事情，一是为林黛玉找到了依靠，二是为林家传承了香火，三是林家的财产也没落到外人手中。林黛玉五岁以前，应该是随林如海在京城，因为那个时候林如海正担任兰台寺大夫。而在京城几年的时间里面，林黛玉却从来没有到过贾府，而偏偏贾敏又把贾府的情况事无巨细的告诉了她，可见贾敏早就在为这个计划做准备了。等贾敏一去世，林黛玉没有了母亲照顾，林如海便把林黛玉送到了贾府，一来是为了让她早早地熟悉环境，二来是为了提前和贾宝玉建立感情基础。这并不是我在信口胡说，我们看贾母对待林黛玉的态度就知道了。按理来说，像贾府这样大的家族，家中的男孩和女孩应该是分院居住的，可贾母偏偏让林黛玉和贾宝玉住到了一起，书中写道：“日则同行同坐，夜则同息同止”。后来搬进了大观园之后，也是林黛玉和贾宝玉住的最近。不仅如此，林黛玉在贾府的待遇标准，一律按照贾宝玉的标准来供应，甚至比迎春、探春、惜春三个亲孙女待遇都高。试问，以贾母的智慧，即便是对林黛玉偏心，怎么会做到如此明目张胆。说白了，这是在为林黛玉树立地位。4根据贾府中其他人的反应可知，贾母和贾敏的这个计划，应该是暗中进行的，也就是说并没有告诉别人。不过， 有两个人应该明显察觉到了贾母的意图，并且作出了完全相反的举动，这两个人分别是邢夫人和王夫人。我们都知道，邢夫人这个人素质并不高，但是在林黛玉初进贾府的时候，表现出来的却是异常的热情。先是主动带着黛玉拜望贾赦，又是殷勤的留黛玉吃饭，最后走的时候亲自送到了仪门前，这自然是在讨好贾母。相反，王夫人的态度则是截然不同。黛玉前往拜望之时，王夫人先是故意冷落黛玉，又是故意把林黛玉往贾政的位子上让，想要看着黛玉出丑，可见对黛玉有着不小的成见。除此以外，王夫人还有一个很奇怪的举动，那就是让黛玉远离贾宝玉。“我不放心的最是一件：我有一个孽根祸胎，原是家里的‘混世魔王’，今日因庙里还愿去了，尚未回来，晚间你看见便知了。你只今后不理睬他，你这些姐妹都不敢沾惹他的。”初看这一段的时候，并没有觉得有什么异常，可仔细琢磨这番话就觉得有些奇怪。因为，王夫人说的这番话，其中大多数信息都是谎话。什么“孽根祸胎”、“混世魔王”，明显是夸张了很多，另外说姐妹们不敢张惹他，后来我们也知道完全不是那么回事。由此来看，王夫人所说的这番话，带有很大的刻意的成分。换句话说，王夫人在提醒林黛玉，离着贾宝玉远一点。说的再通透一点，王夫人知道贾母的意图，她不愿意让贾宝玉娶林黛玉，所以才要给黛玉提这个醒。可黛玉在母亲那里听到的却不是这样，母亲贾敏告诉她的贾宝玉是：“虽极憨顽，在姊妹情中极好的”，事实证明，贾敏所描述的贾宝玉才是更真实的。而王夫人的提醒显然是有些太过了，她接连提醒了黛玉三次，说什么“你别睬他”、“只休信他”，不惜在儿子身上泼脏水，也不想让林黛玉接近他。由此可知，王夫人真正的意图，是破坏贾母的联姻计划。至于因为什么，这个不在我们本篇文章的讨论之列。5在贾府住了四年，林如海身患重病，写信让林黛玉回去。这个时候，贾母的反应非常值得琢磨。“于是贾母定要贾琏送他去，仍叫带回来。”首先，在这里加了“定要”两个字，也就是指明了必须让贾琏去，而且事情办完了以后再把人带回来。如果林黛玉只是寄居在贾府，林如海病故以后，她自然要回去继承家业，怎么能再带回来呢？另外，为什么非要让贾琏送林黛玉回去呢？按照正常的惯例，像这种送人的事情，只需要派几个婆子就可以了，当初接林黛玉的时候就是几个婆子去的。这回非要让贾琏去，意图是很明显的，因为贾琏有办事的能力，而且他是荣国府的长房长孙，他的身份可以代表整个荣国府。所以，这一次送林黛玉回去，并不只是奔丧那么简单，而是要主持料理后事，而后事就包括了林家那巨大财产的归属问题。那么，林家的丧事，作为“外戚”的贾家有什么资格料理呢？如果林黛玉在贾府只是寄居，不管贾府背景实力有多大，他都没有资格料理林家的后事。按照古代的法律来说，如果林如海没有合法继承人的话，林家的财产就要被林氏族人分走。而正是因为林黛玉负有兼祧林家香火的责任， 所以就成了林家合法的继承人，这样一来贾府也就有了处理林家后事的义务，林家的财产作为黛玉的嫁妆也就顺理成章的归到了贾家。事后，贾琏曾经跟王熙凤说过，要是再发个“三二百万银子”的财就好了，这“二三百万银子”的巨款自然就是林家的家产。这个事情办完以后，林黛玉“兼祧”的秘密，自然也就在贾琏夫妻之间公开了。所以，王熙凤在得知消息以后，就对贾宝玉说道：“你林妹妹可在咱们家住长了”。如果你不知道“兼祧”的秘密，像这种细节的描写，你是理解不了的。小说第二十六回，小丫头佳穗对红玉说：“我好造化！才刚在院子里洗东西，宝玉叫往林姑娘那里送茶叶，花大姐姐交给我送去。可巧老太太那里给林姑娘送钱来，正分给他们的丫头们呢。见我去了，林姑娘就抓了两把给我，也不知多少。”贾府中所有人的日常花销，都是靠月例银子，而月例银子是王熙凤负责发的，就连贾宝玉都没有额外的钱，为什么贾母又专程给黛玉送钱呢？原因很简单，因为黛玉的嫁妆是在贾母那里，所以贾母要再单独给她分一份钱。后来，王熙凤在说到林黛玉的嫁妆时，说贾母自会有一份体己钱拿出来，用不到宫中的钱。如果林黛玉是单纯被收养，就应该和其他的姑娘一样，由宫中出钱准备嫁妆，又为什么要贾母自己掏体己钱呢？原因是一样的，因为林黛玉的嫁妆早就准备好了，一直都在贾母那里保管着。而等到黛玉和宝玉结婚的时候，林黛玉的嫁妆由贾母出，贾宝玉的彩礼由宫中出，到时候这两笔巨款都会留给贾宝玉。这也就解释了另外一个问题：为什么贾母不逼着贾宝玉读书求功名？因为求功名并不非得要读书，捐官也是一条路，荫封也是一条路。宝玉的路早就被铺好了，结婚后首先能得到一大笔钱，他们所生的孩子还能继承林家的爵位。小结文章比较长，最后做一个简单的总结：林黛玉之所以进贾府，目的是为了和贾府联姻，这样既能让林黛玉有所依靠，又可以让他的孩子“兼祧”林家的香火。这是作者埋在小说中的一条暗线，贾府中的很多明争暗斗，都是因为这个秘密的联姻引起的，因为这个联姻触动了王夫人、薛姨妈等人的利益。带着这个结论去重新读小说，能看懂很多以前看不懂的细节。【原创】王玄陵",
        # "林黛玉家产去向之谜：世人都冤枉贾府了！	文/夕四少 一直以来，关于林黛玉家产的问题，被红学家乃至红迷们争论不休，莫衷一是，但普遍都认为，林黛玉的家产是被贾府侵吞了，被用来修建省亲别墅了，这个推断对不对呢？我们今天不妨来分析一下。在分析林黛玉家产之前，我们先来看下林家的背景，这一点在原文第二回中有过详细的介绍。原文：原来这林如海之祖，曾袭过列侯，今到如海，业经五世。起初时，只封袭三世，因当今隆恩盛德，远迈前代，额外加恩，至如海之父，又袭了一代；至如海，便从科第出身。虽系钟鼎之家，却亦是书香之族。只可惜这林家支庶不盛，子孙有限，虽有几门，却与如海俱是堂族而已，没甚亲支嫡派的。今如海年已四十，只有一个三岁之子，偏又于去岁死了。虽有几房姬妾，奈他命中无子，亦无可如何之事。这一段文字交代了林家很多信息，首先是林家祖上亦是豪门贵族，到林如海已经是第五代了；其次林家不仅有钱（钟鼎之家），而且是书香门第（书香之族）；再次林如海除了黛玉之母，是有几房姬妾的，且有几门隔亲的堂族。第四个是林如海命中无子，后代只有黛玉一人；第五个即林如海是科甲出身，这一点与贾府中的贾赦、贾政、贾珍、贾琏等世袭官职大不同。关于林如海的官职，原文说：这林如海……乃是前科的探花，今已升至兰台寺大夫，本贯姑苏人氏，今钦点出为巡盐御史，到任方一月有余。我们知道，巡盐御史是明清时期的一个重要官职，那时候盐业掌握在国家手中，禁止民间贩私盐，这一块利益很大，可以说是一个肥缺，皇帝一般只任用身边亲信。历史上，曹雪芹祖父曹寅、舅祖父李煦因深得康熙皇帝信任，都曾任过两淮巡盐史。由林如海的官职我们可知两点，第一，林家深得皇帝信任；第二，这个肥缺油水不会少，也就是说林家不缺钱，甚至可以说富得流油，不然也称不上钟鼎之家，也不会业经五世而不衰。我们再来看贾府涉嫌贪没的证据，被引用最多的证据来自原文第七十二回，这时候贾府已经左支右绌，财政困难，没了现金流，败落之兆已现，从宫里的太监们接二连三变着法儿来贾府索要银子即可知，这一回，贾琏也开始打起了贾母的主意，通过鸳鸯向贾母借当。巧妇难为无米之炊，因为没钱，贾琏由不得叹息道：“这会子再发个三二百万的财就好了。”很多人揪住这个不放，既然贾琏说了“这会子”又说了“再”且还是“三二百万”，也就是说前面他曾发过这样的横财，找来找去，最可疑的就是林黛玉家产的去向，因为林如海病重时，是贾琏带了黛玉回南方并在林如海逝世后带回的，彼时黛玉尚小，也就是说，他最清楚林家财产。但仅凭一句话，就推断为是贾琏贪了林黛玉财产，或贾府贪了林黛玉财产，显得太过武断了些，毕竟曹公没有直接的文字提到林黛玉家产，所以我们看到这里，就不得不解决这样几个问题。第一，虽然林如海任巡盐御史，是钟鼎之家，但毕竟也是经过五代，像贾府这样经过四代，就现了颓相，他真的有三二百万甚至更多的家私吗？第二，林如海有几房姬妾，且有几个堂亲，他病逝后这些人难道会眼睁睁地看着贾琏把所有家产都带走吗？秦钟还没去世的时候，他的亲戚们都已经提前过来准备分家产了，林如海这样的大家族就更是可想而知了，贾琏一人之力，会如此顺利带走那么多钱财吗？第三，以林如海的头脑，他自知病重，为了避免亲族之间争家产，他有没有可能提前写好遗嘱，分配好家产？这样的话，林黛玉自然会得到一笔遗产，这笔遗产会有多少呢？这些钱真的是被贾琏贪了吗？林黛玉是贾母的亲外孙女，三二百万的钱财是个很大的书目，贾琏有这么大的胆子吗？提出这个问题，目的只有两个，那就是要弄清楚林如海死后，到底有没有留下巨额财产？还是并没有什么财产留下？当然，如今学界普遍都认为林如海是留下了巨额遗产的，且都认为这个遗产后来被用在了修建省亲别墅上。我不知道这些人都从哪里推断出来的观点，还是个人合理的臆测？关于修建省亲别墅的花费来源，原文中有过详细交代，“也不过拿着皇帝家的银子往皇帝身上使罢了！”以贾府之贵，元春又被晋封为贵妃，此时的贾府，不太可能连修建省亲别墅的钱都拿不出来，以至于动用了林家遗产，这都是很多人的推断，曹公可没有这么说。我个人认为，修建省亲别墅这样倍有面子倍长脸的事儿，贾府不仅能拿得出这个钱，他也不太可能花一个外姓之人的遗产。无论从哪一方面考量，它都不太可能动用林黛玉继承的财产，再说，这些钱到底有多少，我们并不知道。原文第五十三回，替宁府管理田产地租的管家乌庄头来进贡，说起荣府的开支，认为有万岁爷和贵妃在后面撑着，少不了钱，这时候贾蓉说：“你们山坳海沿子上的人，那里知道这道理。娘娘难道把皇上的库给了我们不成！……这二年那一年不多赔出几千银子来！头一年省亲连盖花园子，你算算那一注共花了多少，就知道了。再两年再一回省亲，只怕就精穷了。”由此可佐证一件事，贾府盖省亲别墅花的是自己的钱，根本没动林黛玉的财产。关于黛玉财产，原文中有过几处明确的描写，细心的读者会发现，林黛玉的生活费不是从公帐上出的，跟三春不一样。原文第二十六回，宝玉院里的小丫头佳蕙跟小红聊天说：“我好造化！才刚在院子里洗东西，宝玉叫往林姑娘那里送茶叶，花大姐姐交给我送去。可巧老太太那里给林姑娘送钱来，正分给他们的丫头们呢。见我去了，林姑娘就抓了两把给我，也不知多少。你替我收着。”由此可知，林黛玉的生活费是贾母每月专门派人送去的，从这一点，我们似乎可以得出一个结论，林黛玉的生活费花的是自己家的钱，是她父亲留给她的遗产，不是贾府的钱。这笔钱有多少呢？原文第四十五回，金兰契互剖金兰语一回，此时的黛玉已经开始向宝钗吐露生活之难。黛玉道：“你如何比我？你又有母亲，又有哥哥，这里又有买卖地土，家里又仍旧有房有地。你不过是亲戚的情分，白住了这里，一应大小事情，又不沾他们一文半个，要走就走了。我是一无所有，吃穿用度，一草一纸，皆是和他们家的姑娘一样，那起小人岂有不多嫌的。”从这段话里，我们似乎又得出了黛玉的另一种处境，即此时的她一无所有，吃穿用度，一草一纸，花的都是贾府的钱，这就有两种可能，一种可能是贾府侵吞了林家遗产，而黛玉并不知情或知情但无可奈何。一个是林如海并没有留给黛玉过多的遗产，哪个可信度更高呢？我更倾向于后者。很多人都觉得林如海是巡盐御史，家里又先后经过五代，攒下来的钱财不说富可敌国，富甲一方应该是有的，但俗语说富不过三代，也许林家的富贵只是因为是官宦之家，是书香之族，未必就一定有大笔大笔的银子，且林如海做巡盐御史并没有多久，就去世了。他如果是个贪官，会得皇帝如此信任吗？说到这，索性再扯开一笔，一直都有人问，既然是四大家族彼此联络有亲，为什么黛玉之母没有嫁给四大家族之人，而是嫁给了林如海？这个问题本身就是无稽之谈！照这么说，李纨、尤氏、邢夫人都不应该嫁进贾府，薛蟠也不应该娶夏金桂，贾蓉也不应该娶秦可卿了，如果都是四大家族联络有亲，那么红楼梦也不会成为经典了。这不过是曹公早已拟定好的人设而已，哪有那么多为什么？话说回来，贾敏既然是贾母最疼爱的女儿，她为自己女儿择偶的标准是什么呢？是一定要有钱吗？关于贾母的择偶标准，原文第二十九回有一段描写。清虚观的张道士要给宝玉说亲，被贾母委婉拒绝了。贾母道：“上回有和尚说了，这孩子命里不该早娶，等再大一大儿再定罢。你可如今打听着，不管他根基富贵，只要模样配的上就好，来告诉我。便是那家子穷，不过给他几两银子罢了。只是模样性格儿难得好的。”为自己的亲孙子选媳妇，贾母的标准不是大富大贵，而是模样好，性格好就行，同理，她为自己女儿选夫婿，最看重的应该也不是大富大贵，而是品行端正，读书上进，为人正派，书香门第之类的条件，这些林如海完全符合，所以，贾敏能嫁给林如海，最主要的原因一定不是因为林家比贾家还富有，而是因为林家深得皇帝恩宠，是比贾府资格还要老大的豪门贵族。综上，林黛玉的家产未必就是被贾府侵吞了，也许她根本没有继承那么多遗产，要么被瓜分了，要么林如海根本没有留下太多遗产。因为曹公没有明确的文字，所以，很多东西，我们只能合理推断，而关于林黛玉家产的问题，我更倾向于贾府并没有侵吞其财产。",

        "晴雯 霁月难逢，彩云易散。心比天高，身为下贱。风流灵巧招人怨。寿天多因诽谤生，多情公子“空牵念”。① 霁月难逢，彩云易散 - “霁月”，明净开朗的境界，旧时称赞人品行高尚，胸怀洒落，就说如“光风霁月” ； 雨后新叫 “霁”，寓“晴”字。“彩云” 喻美好；云呈彩叫 “霁”，这两句说像晴雯这样的人极为难得，因而也就难于为阴暗、污浊的社会所容，她的周围环境正如册子上所画的，只有“满纸乌云而已。”② 心比天高，身为下贱 - 是说晴雯虽身为丫鬟，却从不肯低三下四地奉迎讨好主子，没有阿谀献媚的奴才像，这样的性格是她不幸命运的根源。③ 风流灵巧招人怨 - 传统道德提“女子无才便是德”，要求安分守己，不必风流灵巧，尤其是奴仆，如果模样标致、倔强不驯、必定会招来一些人的嫉恨。④ 寿夭 - 短命夭折。 晴雯被迫害而死时仅16岁。⑤ 多情公子 - 指贾宝玉。撕扇子作千金一笑晴雯给宝玉拿扇子，失手摔坏了扇子，宝玉好心相劝，晴雯却却故意讥讽袭人；袭人来劝和，又遭到晴雯更加犀利的讽刺，以致宝玉要赶走晴雯，后在全体丫鬟和黛玉的劝和下作罢。不久，宝玉游园见到小憩的晴雯，责其与袭人叫板，宝玉、晴雯言归于好，晴雯认为撕扇子有趣，宝玉便赠扇给她撕，觉得用几把扇子换美人一笑十分值得。她不仅撕了一大堆名扇，还将宝玉、麝月的都撕了。勇补雀金袭贾母给了宝玉一件极为珍贵的雀金袭，宝玉穿着它去给舅舅拜寿，不小心把这雀金袭烧了一个顶针大的小洞。麝月就打发老嬷嬷找能工巧匠织补，结果没有一个人敢揽这活，都不认得这是什么袭皮，怎么织补。只有晴雯识得此物，且只有她一人能织补此物。为解宝玉之忧，重病的晴拼命挣扎织补金雀袭，一针一线，一直做到凌晨4点多；当最后一针补好时，只见晴雯“哎呦”了一声，就声不由主睡下了。晴雯之死晴雯在患了重病的情况下，为了宝玉，“病补雀金袭”，加重病情，后又因王善宝家的在王夫人处搬弄是非，使得晴雯被赶出大观园，后忧愤成疾，不久病逝。晴雯之死的重点在一个“屈”字。作者写宝玉去看望晴雯，晴雯悲愤地对宝玉说：“只是一件，我死了也不甘心的，我虽生得比别人略好些，并没有私情蜜勾引你怎样，如何一口死咬定了我是个狐狸精？我太不服，今日既已耽了虚名，而且临死，不是我说一句后悔的话，早知如此，当日也另有个道理”。晴有林风，晴雯性格当中最可宝贵的一面，和黛玉非常相像，就是她有一种比较自觉的人格意识和朦胧的平等意识。曹雪芹在介绍十二钗的册子时，将晴雯置于首位，是有心的安排，是作者的偏爱。",
        "袭人 枉自温柔和顺，空云似桂如兰；堪羡优伶有福，谁知公子无缘。• 枉自温柔和顺 - 指袭人白白地用“温柔和顺”的姿态去博得主子们的好感。• 空云似桂如兰 - “似桂如兰” ，暗点其名。宝玉从宋代陆游《村居书喜》诗“花气袭人知骤暖” （小说中改“骤” 为 “昼” ）中取 “袭人” 二字为她取名，而兰桂最香，所以举此，但“空云” 二字则是对香的否定。• 堪羡优伶有福 - 在这里常用调侃的味道。优伶，旧称唱戏艺人为优伶。这里指蒋玉菡。• 谁知公子无缘 - 作者在八十回后原写袭人在宝玉落到饥寒交迫的境地之前，早已嫁给了蒋玉菡，只留麝月一人在宝玉身边，所以诗的后面两句才这样说。袭人原来出身贫苦，幼小时因为家里没饭吃，老子娘要饿死，为了换得几两银子才卖给贾府当了丫头。可是她在环境影响下所逐渐形成的思想和性格却和晴雯相反。她的所谓“温柔和顺” ，颇与薛宝钗的 “随分从时” 相似，合乎当时的妇道标准和礼法对奴婢的要求。这样的女子，从封建观点看，当然称得上“似桂如兰”。晴雯虽“身为下贱” 却“心比天高”，性如烈火，敢怒敢为，哪怕因此得罪主子，招至大祸也在所不惜。袭人则温顺驯服，并设身处地为主人着想，惟恐不能恪守职任。晴雯原本比袭人起点高，她虽然身世堪怜，十来岁上被卖到赖家，已记不得家乡父母，想来中间不知被转卖了多少道，但因生得伶俐标致，得到贾母喜爱，像个小宠物一样带在身边，稍大又下派到宝玉房里，虽然因资历问题，薪水不如袭人，却是贾母心中准姨娘的重点培养对象，前途相当可观。而袭人自以为是贾母给了宝玉的，贾母对这个丫头并没有多大兴趣，只当她是个锯了嘴的葫芦，不过比一般的丫鬟格外尽心尽力罢了。倘若把晴雯和的袭人人生比喻成一场牌，晴雯的牌明显起的比袭人好，外形才艺都属上乘，还在上级心里挂了号，袭人则一手的小零牌，几乎看不到未来。袭为副钗，袭人个性与宝钗相似，整天劝宝玉读书，学习“仕途经济” ，最终受到王夫人的赏识，却是 “枉自温柔和顺 ，空云似桂如兰”。作者在判词中用“枉自” “空云” “堪羡” “谁知” ，除了暗示她将来的结局与初愿相违外，还带有一定的嘲讽意味。",
        "香菱 根并荷花一茎香，平生遭际实堪伤；自从两地生孤木，致使香魂返故乡。① 根并荷花一茎香 - 暗点其名。香菱本名英莲，莲就是荷，菱与何同生一池，所以说根在一起，书中香菱曾释自己的名字说，“不独菱花，就连荷叶莲蓬都是有一股清香的。” （八十回）② 遭际 - 遭遇③ 自从两地生孤木，致使香魂返故乡 - 这是说自从薛蟠娶夏金桂为妻之后，香菱就被迫害～而死。 “两地生枯木”，两个土字加上一个木字，是金桂的“桂” 字。“魂返故乡”，指死。册上所画也是这个意思。香菱是甄士隐的女儿，她一生的遭遇极不幸，名为甄英莲，其实就是“真应怜”。出生乡宦家庭，3岁即被人偷走，十几岁时被呆霸王薛蟠强买为妾。按照曹雪芹本来的构思，她是被夏金桂迫害而死的。可是，到了程高本序书中却让香菱一直活下去，在第一百零三回中写夏金桂在汤里下毒，谋害香菱，结果反倒毒死了自己，以为只有这样写坏心肠的人的结局，才足以显示 “天理昭彰，自害自身”。 把曹雪芹的意图改变成一个饱含着惩恶扬善教训的离奇故事，实在是弄巧成拙。至于写到夏金桂死后，香菱被扶正，当上正夫人，更是一显然不符曹雪芹的本意的。曹雪芹塑造的香菱，娇憨天真、纯洁温和、得人怜爱。香菱虽遭厄运的磨难，却依然浑融天真，毫无心机，她总是笑嘻嘻地面对人的一切，她恒守着温和专一的性格。当薛蟠在外寻花问柳被人打得臭死，香菱哭得眼睛都肿了，她为自己付出珍贵的痴情。 薛蟠外出做生意，薛宝钗把她带入大观园来住，她有机会结识众姑娘，为了掲示香菱书香人家的气质，曹雪芹还安排了香菱学诗的故事。“香菱拿了诗，回至蘅芜苑中，诸事不顾，只向灯下一首一首的读起来。宝钗连催她数次睡觉，她也不睡。”“如此茶无心，坐卧不定。”“只在池边树下，或坐在山石上出神，或蹲在地下抠土，来往的人都诧异。”“至三更以后上床卧，两眼鳏鳏，直到五更方才朦胧睡去了。”学诗专注投入，乃至痴迷的境界。香菱学诗，她先拜黛玉为师，并在黛玉的指导下细细品味王维的诗，其次是一边读杜甫诗，一边尝试作诗，几经失败，终于成功，梦中得句，写出了 “精华欲掩料应难，影自娟娟魄自寒” “博得嫦娥应借问，缘何不使永团圆” 的精彩诗句，赢得众人赞赏，被补为《海堂诗社》的社员。曹雪芹在百草千花、万紫千红的大观中特意植入一朵暗香的水菱。这时香菱短暂的温馨画面，给了读者一份小小的安慰。",
        
        "贾宝玉 贾宝玉奉命应写 《有凤来仪》《蘅芷清芬》《怡红快绿》《杏帘在望》四首。其中《杏帘在望》 是林黛玉帮他写的。（因见宝玉大费神思，为省他精神而代作，元春认为代作这首是三首之冠）《怡红快绿》中的“绿玉” 改为 “绿蜡”，是宝钗提醒的。（因宝钗揣测出元春不喜 “绿玉”。偏是她才会如此留心啊。）“宝玉挨打”后宝钗与黛玉的不同探望方式（34回）宝钗托药而来，（光明正大之态，意欲让大家注意到她对宝玉的关切心思） 流露真情时懂得自我控制，内敛而不外露。对于宝玉挨打，她以为事出有因，并借机规劝宝玉走仕途经济之路。黛玉在无人时悄悄来看宝玉，她的深情表现在她的无声之泣及简单的言辞里。写 “桃儿一般的” 眼睛，可见哭泣时间之长与伤心之。但又极不愿别人看到她对宝玉的关心。黛玉的关切完全是真情流露，相比之下，宝钗关切多半是表面文章。黛玉在人生观上与宝玉相同，因为她能抛弃世俗的功利，她看宝玉，送去的是一颗真心，她从不说“那些混帐话”，也不劝宝玉走仕途之路，既便她叫宝玉 “你从此可都改了罢”，也是为宝玉的安危着想。林黛玉与薛宝钗，一个是寄人篱下的孤女，一个是皇家大商人的千金； 一个天真率直，一个城府极深；一个孤立无援，一个有多方支持；一个作叛逆者知己，一个为卫道而说教 。脂砚斋曾有过“钗黛合一” 说，作者将她俩三首诗中并提，除了因为她们在小说中的地位相当外，至少还可以通过贾宝玉对她们的不同态度的比较，以显示钗黛的命运遭遇虽则不同，其结果都是一场悲剧。“对作者来说，或许人世间的美好幸福是不能全得的。 有所取，就有所舍；有所得，就有所失。林黛玉和薛宝钗各有千秋，好像两人合在一起才最完美。如果他们是两个人，就永远不完美。在作者幻想的世界里，在判词当中，她们变成了合在一起的生命形态。”",
        
        "薛宝钗 可叹停机德，堪怜咏絮才!玉带林中挂，金簪雪里埋。①可叹停机德 - 据说薛宝钗，意思是虽然有着合乎孔孟之道标准的那种贤妻良母的品德，可惜徒劳无功。 “停机德”，出《后汉书 • 列女传 • 乐羊子妻》。 故事说： 乐羊子远出寻师求学，因为想家，只过了一年就回家了。他的妻子就拿刀割断了织布机上的绢，以此来比学业中断，规劝他继续求学，谋取功名，不要半途而废。④ 金簪雪里埋 - 这句说薛宝钗，前三字暗点其名； “雪” 谐 “薛”，金簪比 “宝钗”。本是光耀头面的首，竟埋在寒冷的雪堆里，这是对一心想当宝二奶奶的薛宝钗的冷落处境的写照。【花名签酒令八首之】牡丹 - 艳冠群芳（宝钗）• 任是无情也动人（牡丹 端庄、典雅，象征富贵，有大家闺秀之感，薛宝钗就和牡丹一样，大方端庄，追求富贵。花签上的诗句切合宝钗性情冷淡而又能处处得人好的性格特点。）临江仙（薛宝钗的柳絮词）白玉堂前春解舞，东风卷得均匀，蜂围蝶阵乱纷纷。几曾随逝水？岂必委芳尘？万缕千丝终不改，任他随聚随分。 韶华休笑本无根。好风凭借力，送我上青云。   18回中贾妃在游大观园时命贾宝玉等众兄妹各题一匾一诗，贾宝玉奉命应写四首，其中有一首是林黛玉帮他写的，有一首是宝钗提醒他修改的。终生误（宝钗）都道是金玉良姻，俺只念木石前盟。空对着，山中高士晶莹雪； 终不忘，世外仙姝寂寞林。叹人间，美中不足今方信： 纵然是，齐眉举案，到底意难平。这首曲子写贾宝玉婚后仍不忘怀死去的林黛玉，写薛宝钗徒有 “金玉良姻” 的虚名而实际上则终身寂寞。曲名《终身误》就包含这个意思。",
        "林黛玉 林黛玉是和贾宝玉关系最亲密的人。可叹停机德，堪怜咏絮才!玉带林中挂，金簪雪里埋。② 堪怜咏絮才 - 这句说林黛玉，意思说如此聪明有才华的女子，她的命运值得同情。 “咏絮才”，用晋代谢道韫的故事： 有一次，天下大雪，谢道韫的叔父谢安对雪吟句说 “白雪纷纷何所似？” 道韫的哥哥谢朗答道： “撒盐空中差可拟” 谢道韫接着说： “未若柳絮因风起” 谢安一听大为赞赏。见 《世说新语》。③ 玉带林中挂 - 这句说林黛玉，前三字倒读即谐其名。从册里的画 “两株枯木（双木为“林” ），木上悬着一围玉带看，可能又寓宝玉 “悬” 念 “挂”，牵挂死去的黛玉的意思。【花名签酒令八首之】芙蓉 - 风露清愁（黛玉）• 莫怨东风当自嗟（芙蓉清丽、纯洁动人，作者将林黛玉喻作芙蓉，暗指她有出水芙蓉般纯真的天性，美丽、孤傲。禁不起 “狂风” 摧折，亦即暗示她后来受不了贾府事变 “狂风” 的袭击，终于泪尽而逝。“当自嗟”，说明作者固然同情黛玉的不幸，但也深深惋惜她过于脆弱。望凝眉一个是阆苑仙葩，一个是美玉无瑕。若说没奇缘，今生偏又遇着他； 若说有奇缘，如何心事终虚化？ 一个枉自嗟呀，一个空劳牵挂。一个是水中月，一个是镜中花。想眼中能有多少泪珠儿，怎禁得秋流到冬尽、 春流到复！这首曲子写宝、黛的爱情理想因变故而破灭，写林黛玉的泪尽而逝，曲名《枉凝眉》，意思是悲愁有何用？也即曲中所说的 “枉自嗟呀” 。 凝眉，皱眉，悲愁的样子。宝玉曾赠黛玉表字 “颦颦”。黛玉之死第九十六回 瞒消息凤姐设奇谋，泄机关颦儿迷本性。第九十七回 林黛玉焚稿断痴情，薛宝钗出闺成大礼。第九十八回 苦绛珠魂归离恨天，病神瑛泪洒相思地。黛玉弥留之际，直声叫道： “宝玉，宝玉，你好…” 有无限未尽之意。宝玉，宝玉，你好糊涂！宝玉，宝玉，你好狠心！宝玉，宝玉，你好绝情！宝玉，宝玉，你好自为之！宝玉，宝玉，你好好保重…",
        "贾元春 画： 一张弓，弓上挂着一个香橼。（画中画的似乎与宫闱事有关，因为“弓 ” 可谓 “宫”。“ 橼 ” 可谐 “元”。）二十年来辨是非，榴花开处照宫闱。三春争及初春景，虎兔相逢大梦归。",
        "贾探春 才自清明志自高，生于末世运偏消；清明涕泣江边望，千里东风一梦遥。",
        "史湘云 富贵又何为，襁褓之间父母违；展眼吊斜晖，湘江水逝楚云飞。",
        "妙玉 欲洁何曾洁，云空未必空；可怜金玉质，终陷淖泥中。",
        "贾迎春 子系中山狼，得志便猖狂；金闺花柳质，一载赴黄粱。",
        "贾惜春 勘破三春景不长，缁衣顿改昔年妆；可怜绣户侯门女，独卧青灯古佛旁。",
        "王熙凤 凡鸟偏从末世来，都知爱慕此生才；一从二令三人木，哭向金陵事更哀。",
        "贾巧姐 势败休云贵，家亡莫论亲；偶因济村妇，巧得遇恩人。",
        "李纨 桃李春风结子完，到头谁似一盆兰；如冰水好空相妒，枉与他人作笑谈。",
        "秦可卿 - 在现实中引领宝玉进入梦境的人。袅娜柔情。情天情海幻情身，情既相逢必主淫；漫言不肖皆荣出，造衅开端实在宁。",
    ]

# queries = ["When was artificial intelligence founded", 
#            "Where was Alan Turing born?"]
# query_embeddings = embedding_fn.encode_queries(queries)
# print("Embeddings:", query_embeddings)
# print("Dim", embedding_fn.dim, query_embeddings[0].shape)

# 1 Embedding Docs
docs_embeddings = embedding_fn.encode_documents(docs)
print("Embeddings:", docs_embeddings)
print("Dim:", embedding_fn.dim, docs_embeddings[0].shape)

# 2 Build Data 
# Each entity has id, vector representation, raw text, and a subject label to filtering metadata.
data = [
    {"id": i, "vector": docs_embeddings[i], "text": docs[i], "subject": "文学评论", "metadata": {"author": "曹雪芹"}}
    for i in range(len(docs_embeddings))
]
# print("Data has", len(data), "entities, each with fields: ", data[0].keys())
# print("Vector dim:", len(data[0]["docs_embeddings"]))

# 3 Create db
if client.has_collection(collection_name="literature"):
    client.drop_collection(collection_name="literature")
client.create_collection(
    collection_name="literature",
    dimension=DIM,  # The vectors we will use in this demo has 768 dimensions
)

# 4 Insert Data
res = client.upsert(collection_name="literature", data=data)
# print(res)

# 5 search
# Embedding Query
query_vectors = embedding_fn.encode_queries(["元春和迎春的关系如何"])
res = client.search(
    collection_name="literature",  # target collection
    data=query_vectors,  # query vectors
    limit=4,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)
print(res)