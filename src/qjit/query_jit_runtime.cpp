#include "qjit/query_jit_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "qjit/query_jit_scheduler.h"

alignas(4096) const uint16_t qjit_bloom_masks[2048] = {
   15, 23, 27, 29, 30, 39, 43, 45, 46, 51, 53, 54, 57, 58, 60, 71, 75, 77, 78, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 113, 114, 116, 120, 135, 139, 141, 142, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 177, 178, 180, 184, 195, 197, 198, 201, 202, 204, 209, 210, 212, 216, 225, 226, 228, 232, 240, 263, 267, 269, 270, 275, 277, 278, 281, 282, 284, 291, 293, 294, 297, 298, 300, 305, 306, 308, 312, 323, 325, 326, 329, 330, 332, 337, 338, 340, 344, 353, 354, 356, 360, 368, 387, 389, 390, 393, 394, 396, 401, 402, 404, 408, 417, 418, 420, 424, 432, 449, 450, 452, 456, 464, 480, 519, 523,
   525, 526, 531, 533, 534, 537, 538, 540, 547, 549, 550, 553, 554, 556, 561, 562, 564, 568, 579, 581, 582, 585, 586, 588, 593, 594, 596, 600, 609, 610, 612, 616, 624, 643, 645, 646, 649, 650, 652, 657, 658, 660, 664, 673, 674, 676, 680, 688, 705, 706, 708, 712, 720, 736, 771, 773, 774, 777, 778, 780, 785, 786, 788, 792, 801, 802, 804, 808, 816, 833, 834, 836, 840, 848, 864, 897, 898, 900, 904, 912, 928, 960, 1031, 1035, 1037, 1038, 1043, 1045, 1046, 1049, 1050, 1052, 1059, 1061, 1062, 1065, 1066, 1068, 1073, 1074, 1076, 1080, 1091, 1093, 1094, 1097, 1098, 1100, 1105, 1106, 1108, 1112, 1121, 1122, 1124, 1128, 1136, 1155, 1157, 1158, 1161, 1162, 1164, 1169, 1170, 1172, 1176, 1185,
   1186, 1188, 1192, 1200, 1217, 1218, 1220, 1224, 1232, 1248, 1283, 1285, 1286, 1289, 1290, 1292, 1297, 1298, 1300, 1304, 1313, 1314, 1316, 1320, 1328, 1345, 1346, 1348, 1352, 1360, 1376, 1409, 1410, 1412, 1416, 1424, 1440, 1472, 1539, 1541, 1542, 1545, 1546, 1548, 1553, 1554, 1556, 1560, 1569, 1570, 1572, 1576, 1584, 1601, 1602, 1604, 1608, 1616, 1632, 1665, 1666, 1668, 1672, 1680, 1696, 1728, 1793, 1794, 1796, 1800, 1808, 1824, 1856, 1920, 2055, 2059, 2061, 2062, 2067, 2069, 2070, 2073, 2074, 2076, 2083, 2085, 2086, 2089, 2090, 2092, 2097, 2098, 2100, 2104, 2115, 2117, 2118, 2121, 2122, 2124, 2129, 2130, 2132, 2136, 2145, 2146, 2148, 2152, 2160, 2179, 2181, 2182, 2185, 2186, 2188, 2193, 2194, 2196, 2200, 2209, 2210, 2212, 2216, 2224, 2241, 2242, 2244, 2248,
   2256, 2272, 2307, 2309, 2310, 2313, 2314, 2316, 2321, 2322, 2324, 2328, 2337, 2338, 2340, 2344, 2352, 2369, 2370, 2372, 2376, 2384, 2400, 2433, 2434, 2436, 2440, 2448, 2464, 2496, 2563, 2565, 2566, 2569, 2570, 2572, 2577, 2578, 2580, 2584, 2593, 2594, 2596, 2600, 2608, 2625, 2626, 2628, 2632, 2640, 2656, 2689, 2690, 2692, 2696, 2704, 2720, 2752, 2817, 2818, 2820, 2824, 2832, 2848, 2880, 2944, 3075, 3077, 3078, 3081, 3082, 3084, 3089, 3090, 3092, 3096, 3105, 3106, 3108, 3112, 3120, 3137, 3138, 3140, 3144, 3152, 3168, 3201, 3202, 3204, 3208, 3216, 3232, 3264, 3329, 3330, 3332, 3336, 3344, 3360, 3392, 3456, 3585, 3586, 3588, 3592, 3600, 3616, 3648, 3712, 3840, 4103, 4107, 4109, 4110, 4115, 4117, 4118, 4121, 4122, 4124, 4131, 4133, 4134, 4137, 4138, 4140, 4145,
   4146, 4148, 4152, 4163, 4165, 4166, 4169, 4170, 4172, 4177, 4178, 4180, 4184, 4193, 4194, 4196, 4200, 4208, 4227, 4229, 4230, 4233, 4234, 4236, 4241, 4242, 4244, 4248, 4257, 4258, 4260, 4264, 4272, 4289, 4290, 4292, 4296, 4304, 4320, 4355, 4357, 4358, 4361, 4362, 4364, 4369, 4370, 4372, 4376, 4385, 4386, 4388, 4392, 4400, 4417, 4418, 4420, 4424, 4432, 4448, 4481, 4482, 4484, 4488, 4496, 4512, 4544, 4611, 4613, 4614, 4617, 4618, 4620, 4625, 4626, 4628, 4632, 4641, 4642, 4644, 4648, 4656, 4673, 4674, 4676, 4680, 4688, 4704, 4737, 4738, 4740, 4744, 4752, 4768, 4800, 4865, 4866, 4868, 4872, 4880, 4896, 4928, 4992, 5123, 5125, 5126, 5129, 5130, 5132, 5137, 5138, 5140, 5144, 5153, 5154, 5156, 5160, 5168, 5185, 5186, 5188, 5192, 5200, 5216, 5249, 5250, 5252, 5256,
   5264, 5280, 5312, 5377, 5378, 5380, 5384, 5392, 5408, 5440, 5504, 5633, 5634, 5636, 5640, 5648, 5664, 5696, 5760, 5888, 6147, 6149, 6150, 6153, 6154, 6156, 6161, 6162, 6164, 6168, 6177, 6178, 6180, 6184, 6192, 6209, 6210, 6212, 6216, 6224, 6240, 6273, 6274, 6276, 6280, 6288, 6304, 6336, 6401, 6402, 6404, 6408, 6416, 6432, 6464, 6528, 6657, 6658, 6660, 6664, 6672, 6688, 6720, 6784, 6912, 7169, 7170, 7172, 7176, 7184, 7200, 7232, 7296, 7424, 7680, 8199, 8203, 8205, 8206, 8211, 8213, 8214, 8217, 8218, 8220, 8227, 8229, 8230, 8233, 8234, 8236, 8241, 8242, 8244, 8248, 8259, 8261, 8262, 8265, 8266, 8268, 8273, 8274, 8276, 8280, 8289, 8290, 8292, 8296, 8304, 8323, 8325, 8326, 8329, 8330, 8332, 8337, 8338, 8340, 8344, 8353, 8354, 8356, 8360, 8368, 8385, 8386, 8388,
   8392, 8400, 8416, 8451, 8453, 8454, 8457, 8458, 8460, 8465, 8466, 8468, 8472, 8481, 8482, 8484, 8488, 8496, 8513, 8514, 8516, 8520, 8528, 8544, 8577, 8578, 8580, 8584, 8592, 8608, 8640, 8707, 8709, 8710, 8713, 8714, 8716, 8721, 8722, 8724, 8728, 8737, 8738, 8740, 8744, 8752, 8769, 8770, 8772, 8776, 8784, 8800, 8833, 8834, 8836, 8840, 8848, 8864, 8896, 8961, 8962, 8964, 8968, 8976, 8992, 9024, 9088, 9219, 9221, 9222, 9225, 9226, 9228, 9233, 9234, 9236, 9240, 9249, 9250, 9252, 9256, 9264, 9281, 9282, 9284, 9288, 9296, 9312, 9345, 9346, 9348, 9352, 9360, 9376, 9408, 9473, 9474, 9476, 9480, 9488, 9504, 9536, 9600, 9729, 9730, 9732, 9736, 9744, 9760, 9792, 9856, 9984, 10243, 10245, 10246, 10249, 10250, 10252, 10257, 10258, 10260, 10264, 10273, 10274, 10276, 10280, 10288, 10305,
   10306, 10308, 10312, 10320, 10336, 10369, 10370, 10372, 10376, 10384, 10400, 10432, 10497, 10498, 10500, 10504, 10512, 10528, 10560, 10624, 10753, 10754, 10756, 10760, 10768, 10784, 10816, 10880, 11008, 11265, 11266, 11268, 11272, 11280, 11296, 11328, 11392, 11520, 11776, 12291, 12293, 12294, 12297, 12298, 12300, 12305, 12306, 12308, 12312, 12321, 12322, 12324, 12328, 12336, 12353, 12354, 12356, 12360, 12368, 12384, 12417, 12418, 12420, 12424, 12432, 12448, 12480, 12545, 12546, 12548, 12552, 12560, 12576, 12608, 12672, 12801, 12802, 12804, 12808, 12816, 12832, 12864, 12928, 13056, 13313, 13314, 13316, 13320, 13328, 13344, 13376, 13440, 13568, 13824, 14337, 14338, 14340, 14344, 14352, 14368, 14400, 14464, 14592, 14848, 15360, 16391, 16395, 16397, 16398, 16403, 16405, 16406, 16409, 16410, 16412, 16419, 16421, 16422, 16425, 16426, 16428, 16433, 16434, 16436, 16440, 16451, 16453, 16454,
   16457, 16458, 16460, 16465, 16466, 16468, 16472, 16481, 16482, 16484, 16488, 16496, 16515, 16517, 16518, 16521, 16522, 16524, 16529, 16530, 16532, 16536, 16545, 16546, 16548, 16552, 16560, 16577, 16578, 16580, 16584, 16592, 16608, 16643, 16645, 16646, 16649, 16650, 16652, 16657, 16658, 16660, 16664, 16673, 16674, 16676, 16680, 16688, 16705, 16706, 16708, 16712, 16720, 16736, 16769, 16770, 16772, 16776, 16784, 16800, 16832, 16899, 16901, 16902, 16905, 16906, 16908, 16913, 16914, 16916, 16920, 16929, 16930, 16932, 16936, 16944, 16961, 16962, 16964, 16968, 16976, 16992, 17025, 17026, 17028, 17032, 17040, 17056, 17088, 17153, 17154, 17156, 17160, 17168, 17184, 17216, 17280, 17411, 17413, 17414, 17417, 17418, 17420, 17425, 17426, 17428, 17432, 17441, 17442, 17444, 17448, 17456, 17473, 17474, 17476, 17480, 17488, 17504, 17537, 17538, 17540, 17544, 17552, 17568, 17600, 17665, 17666, 17668,
   17672, 17680, 17696, 17728, 17792, 17921, 17922, 17924, 17928, 17936, 17952, 17984, 18048, 18176, 18435, 18437, 18438, 18441, 18442, 18444, 18449, 18450, 18452, 18456, 18465, 18466, 18468, 18472, 18480, 18497, 18498, 18500, 18504, 18512, 18528, 18561, 18562, 18564, 18568, 18576, 18592, 18624, 18689, 18690, 18692, 18696, 18704, 18720, 18752, 18816, 18945, 18946, 18948, 18952, 18960, 18976, 19008, 19072, 19200, 19457, 19458, 19460, 19464, 19472, 19488, 19520, 19584, 19712, 19968, 20483, 20485, 20486, 20489, 20490, 20492, 20497, 20498, 20500, 20504, 20513, 20514, 20516, 20520, 20528, 20545, 20546, 20548, 20552, 20560, 20576, 20609, 20610, 20612, 20616, 20624, 20640, 20672, 20737, 20738, 20740, 20744, 20752, 20768, 20800, 20864, 20993, 20994, 20996, 21000, 21008, 21024, 21056, 21120, 21248, 21505, 21506, 21508, 21512, 21520, 21536, 21568, 21632, 21760, 22016, 22529, 22530, 22532, 22536,
   22544, 22560, 22592, 22656, 22784, 23040, 23552, 24579, 24581, 24582, 24585, 24586, 24588, 24593, 24594, 24596, 24600, 24609, 24610, 24612, 24616, 24624, 24641, 24642, 24644, 24648, 24656, 24672, 24705, 24706, 24708, 24712, 24720, 24736, 24768, 24833, 24834, 24836, 24840, 24848, 24864, 24896, 24960, 25089, 25090, 25092, 25096, 25104, 25120, 25152, 25216, 25344, 25601, 25602, 25604, 25608, 25616, 25632, 25664, 25728, 25856, 26112, 26625, 26626, 26628, 26632, 26640, 26656, 26688, 26752, 26880, 27136, 27648, 28673, 28674, 28676, 28680, 28688, 28704, 28736, 28800, 28928, 29184, 29696, 30720, 32775, 32779, 32781, 32782, 32787, 32789, 32790, 32793, 32794, 32796, 32803, 32805, 32806, 32809, 32810, 32812, 32817, 32818, 32820, 32824, 32835, 32837, 32838, 32841, 32842, 32844, 32849, 32850, 32852, 32856, 32865, 32866, 32868, 32872, 32880, 32899, 32901, 32902, 32905, 32906, 32908, 32913, 32914,
   32916, 32920, 32929, 32930, 32932, 32936, 32944, 32961, 32962, 32964, 32968, 32976, 32992, 33027, 33029, 33030, 33033, 33034, 33036, 33041, 33042, 33044, 33048, 33057, 33058, 33060, 33064, 33072, 33089, 33090, 33092, 33096, 33104, 33120, 33153, 33154, 33156, 33160, 33168, 33184, 33216, 33283, 33285, 33286, 33289, 33290, 33292, 33297, 33298, 33300, 33304, 33313, 33314, 33316, 33320, 33328, 33345, 33346, 33348, 33352, 33360, 33376, 33409, 33410, 33412, 33416, 33424, 33440, 33472, 33537, 33538, 33540, 33544, 33552, 33568, 33600, 33664, 33795, 33797, 33798, 33801, 33802, 33804, 33809, 33810, 33812, 33816, 33825, 33826, 33828, 33832, 33840, 33857, 33858, 33860, 33864, 33872, 33888, 33921, 33922, 33924, 33928, 33936, 33952, 33984, 34049, 34050, 34052, 34056, 34064, 34080, 34112, 34176, 34305, 34306, 34308, 34312, 34320, 34336, 34368, 34432, 34560, 34819, 34821, 34822, 34825, 34826, 34828,
   34833, 34834, 34836, 34840, 34849, 34850, 34852, 34856, 34864, 34881, 34882, 34884, 34888, 34896, 34912, 34945, 34946, 34948, 34952, 34960, 34976, 35008, 35073, 35074, 35076, 35080, 35088, 35104, 35136, 35200, 35329, 35330, 35332, 35336, 35344, 35360, 35392, 35456, 35584, 35841, 35842, 35844, 35848, 35856, 35872, 35904, 35968, 36096, 36352, 36867, 36869, 36870, 36873, 36874, 36876, 36881, 36882, 36884, 36888, 36897, 36898, 36900, 36904, 36912, 36929, 36930, 36932, 36936, 36944, 36960, 36993, 36994, 36996, 37000, 37008, 37024, 37056, 37121, 37122, 37124, 37128, 37136, 37152, 37184, 37248, 37377, 37378, 37380, 37384, 37392, 37408, 37440, 37504, 37632, 37889, 37890, 37892, 37896, 37904, 37920, 37952, 38016, 38144, 38400, 38913, 38914, 38916, 38920, 38928, 38944, 38976, 39040, 39168, 39424, 39936, 40963, 40965, 40966, 40969, 40970, 40972, 40977, 40978, 40980, 40984, 40993, 40994, 40996,
   41000, 41008, 41025, 41026, 41028, 41032, 41040, 41056, 41089, 41090, 41092, 41096, 41104, 41120, 41152, 41217, 41218, 41220, 41224, 41232, 41248, 41280, 41344, 41473, 41474, 41476, 41480, 41488, 41504, 41536, 41600, 41728, 41985, 41986, 41988, 41992, 42000, 42016, 42048, 42112, 42240, 42496, 43009, 43010, 43012, 43016, 43024, 43040, 43072, 43136, 43264, 43520, 44032, 45057, 45058, 45060, 45064, 45072, 45088, 45120, 45184, 45312, 45568, 46080, 47104, 49155, 49157, 49158, 49161, 49162, 49164, 49169, 49170, 49172, 49176, 49185, 49186, 49188, 49192, 49200, 49217, 49218, 49220, 49224, 49232, 49248, 49281, 49282, 49284, 49288, 49296, 49312, 49344, 49409, 49410, 49412, 49416, 49424, 49440, 49472, 49536, 49665, 49666, 49668, 49672, 49680, 49696, 49728, 49792, 49920, 50177, 50178, 50180, 50184, 50192, 50208, 50240, 50304, 50432, 50688, 51201, 51202, 51204, 51208, 51216, 51232, 51264, 51328,
   51456, 51712, 52224, 53249, 53250, 53252, 53256, 53264, 53280, 53312, 53376, 53504, 53760, 54272, 55296, 57345, 57346, 57348, 57352, 57360, 57376, 57408, 57472, 57600, 57856, 58368, 59392, 61440,
   75, 83, 90, 99, 116, 135, 139, 172, 284, 298, 464, 547, 556, 564, 582, 594, 657, 658, 705, 771, 780, 928, 960, 1045, 1098, 1158, 1176, 1185, 1186, 1283, 1346, 1424, 1576, 2059, 2067, 2083, 2085, 2089, 2115, 2181, 2186, 2188, 2194, 2310, 2369, 2372, 2632, 3105, 3106, 3152, 3392, 3840, 4121, 4138, 4140, 4152, 4208, 4234, 4241, 4362, 4364, 4418, 4488, 4613, 4617, 4674, 4744, 4752, 4992, 5123, 5126, 5132, 5138, 5140, 5256, 6210, 6276, 7176, 8214, 8218, 8233, 8248, 8329, 8337, 8340, 8344, 8360, 8386, 8392, 8416, 8457, 8472, 8488, 8592, 9352, 9476, 9600, 9736, 9744, 9792, 10336, 10376, 10500, 10512, 10754, 11266, 12291, 12300, 12306, 12308, 12322, 12324, 12420, 12432, 12560, 13314, 13316, 13440, 13824, 16397, 16453, 16529, 16580, 16688, 16906, 16929, 16962, 16968,
   17040, 17160, 17411, 17414, 17428, 17448, 17473, 17480, 17504, 17538, 17544, 17665, 17680, 17928, 17984, 18512, 18696, 18945, 19008, 19464, 19488, 20497, 20500, 20513, 20546, 20548, 20612, 20624, 22530, 24593, 24612, 24644, 24706, 25089, 25092, 25096, 25104, 32779, 32789, 32794, 32824, 32850, 32905, 32914, 32929, 32962, 32976, 33060, 33410, 33424, 33664, 33825, 33921, 33922, 33936, 34052, 34080, 34308, 34822, 34828, 34856, 34881, 34884, 34945, 36096, 36884, 36898, 36932, 37056, 37184, 37377, 38016, 38914, 38928, 41120, 41152, 41600, 42240, 43520, 45312, 49155, 49157, 49169, 49172, 49186, 49296, 49410, 49536, 50178, 50184, 50192, 52224, 53280, 54272, 55296, 57352, 57360, 57600, 57856, 59392};

namespace qjit {

// ---------------------------------------------------------------------------
// QjitString
// ---------------------------------------------------------------------------

QjitString MakeString(const char *data, uint32_t len) {
  QjitString s;
  std::memset(&s, 0, sizeof(s));
  if (len <= QJIT_STRING_INLINE_LEN) {
    s.inlined.length = len;
    if (len)
      std::memcpy(s.inlined.inlined, data, len);
  } else {
    s.pointer.length = len;
    std::memcpy(s.pointer.prefix, data, 4);
    s.pointer.ptr = data;
  }
  return s;
}

// ---------------------------------------------------------------------------
// QjitBuffer
// ---------------------------------------------------------------------------

uint8_t *QjitBuffer::Allocate(uint64_t bytes) {
  if (chunks_.empty() || chunks_.back().used + bytes > chunks_.back().capacity) {
    uint64_t cap = next_capacity_;
    if (cap < bytes)
      cap = bytes;
    Chunk c;
    c.data.reset(new uint8_t[cap]);
    c.capacity = cap;
    chunks_.push_back(std::move(c));
    // 1.2x growth (lingo-db GrowingBuffer model)
    next_capacity_ = cap + cap / 5;
  }
  Chunk &c = chunks_.back();
  uint8_t *p = c.data.get() + c.used;
  c.used += bytes;
  total_size_ += bytes;
  return p;
}

// ---------------------------------------------------------------------------
// QjitStringArena
// ---------------------------------------------------------------------------

QjitString QjitStringArena::Copy(const QjitString &src) {
  return Copy(StringData(src), StringLen(src));
}

QjitString QjitStringArena::Copy(const char *data, uint32_t len) {
  if (len <= QJIT_STRING_INLINE_LEN)
    return MakeString(data, len); // inline: copied by value, no arena bytes
  char *dst = reinterpret_cast<char *>(bytes_.Allocate(len));
  std::memcpy(dst, data, len);
  return MakeString(dst, len);
}

// ---------------------------------------------------------------------------
// QjitHashTable
// ---------------------------------------------------------------------------

QjitHashTable::QjitHashTable(uint32_t tuple_size, uint32_t num_workers,
                             int64_t key0_offset)
    : tuple_size_(tuple_size),
      // Keep entries 8-byte aligned: header is 16 bytes, round the row up.
      entry_stride_(sizeof(Entry) + ((uint64_t(tuple_size) + 7) & ~uint64_t(7))),
      key0_offset_(key0_offset), fragments_(num_workers),
      arenas_(num_workers) {}

uint8_t *QjitHashTable::AppendRow(uint32_t worker_id, uint64_t hash) {
  assert(!finalized_);
  Fragment &f = fragments_[worker_id];
  auto *e = reinterpret_cast<Entry *>(f.buffer.Allocate(entry_stride_));
  e->next = nullptr;
  e->hash = hash;
  f.count++;
  return e->Row();
}

uint64_t QjitHashTable::NumEntries() const {
  uint64_t n = 0;
  for (const auto &f : fragments_)
    n += f.count;
  return n;
}

void QjitHashTable::Finalize(QjitWorkerPool *pool) {
  if (finalized_)
    return;
  static const bool trace = std::getenv("AQP_QJIT_FINALIZE_TRACE") != nullptr;
  auto t0 = trace ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};

  uint64_t n = NumEntries();
  uint64_t dir_size = 64;
  while (dir_size < 2 * n)
    dir_size <<= 1;
  directory_.reset(new std::atomic<uintptr_t>[dir_size]);
  dir_mask_ = dir_size - 1;

  struct Range {
    uint8_t *base;
    uint64_t first;
    uint64_t count;
  };
  std::vector<Range> ranges;
  uint64_t total = 0;
  for (auto &f : fragments_) {
    for (const auto &chunk : f.buffer.Chunks()) {
      uint64_t cnt = chunk.used / entry_stride_;
      if (cnt) {
        ranges.push_back({chunk.data.get(), total, cnt});
        total += cnt;
      }
    }
  }

  auto for_each_entry = [&](uint64_t begin, uint64_t end, auto &&fn) {
    size_t r = std::upper_bound(ranges.begin(), ranges.end(), begin,
                                [](uint64_t v, const Range &rg) {
                                  return v < rg.first;
                                }) -
               ranges.begin() - 1;
    for (uint64_t i = begin; i < end; r++) {
      const Range &rg = ranges[r];
      uint64_t off = i - rg.first;
      uint64_t stop = std::min(end - rg.first, rg.count);
      for (; off < stop; off++) {
        auto *e = reinterpret_cast<Entry *>(rg.base + off * entry_stride_);
        fn(e);
      }
      i = rg.first + stop;
    }
  };

  auto zero_dir = [&](uint64_t b, uint64_t e) {
    std::memset(static_cast<void *>(directory_.get() + b), 0,
                (e - b) * sizeof(std::atomic<uintptr_t>));
  };

  constexpr uint64_t kParallelMin = 4096;
  bool parallel = pool && pool->NumWorkers() > 1 && total >= kParallelMin;

  if (parallel) {
    pool->ParallelFor(dir_size, uint64_t(1) << 18,
                      [&](uint64_t b, uint64_t e, uint32_t) { zero_dir(b, e); });
    uint64_t morsel = total / (uint64_t(pool->NumWorkers()) * 8);
    if (morsel < 1024)
      morsel = 1024;

    // Prefetch directory slots ahead to hide DRAM latency.
    // Entries are sequential (cache-friendly), but directory[hash & mask]
    // is random (64MB+ for large HTs → guaranteed LLC miss).
    constexpr int kPFDist = 16;
    struct PFSlot { uint64_t hash; Entry *entry; };

    auto link = [&](uint64_t begin, uint64_t end) {
      int64_t lmin = INT64_MAX, lmax = INT64_MIN;

      PFSlot ring[kPFDist];
      int head = 0, count = 0;

      auto drain_one = [&]() {
        PFSlot &s = ring[head % kPFDist];
        std::atomic<uintptr_t> &dir = directory_[s.hash & dir_mask_];
        uintptr_t old = dir.load(std::memory_order_relaxed);
        do {
          s.entry->next = static_cast<Entry *>(bt_decode(old));
        } while (!dir.compare_exchange_weak(
            old, bt_encode(s.entry, old, s.hash),
            std::memory_order_release, std::memory_order_relaxed));
        head++;
        count--;
      };

      for_each_entry(begin, end, [&](Entry *e) {
        if (key0_offset_ >= 0) {
          int64_t k;
          std::memcpy(&k, e->Row() + key0_offset_, sizeof(k));
          if (k < lmin) lmin = k;
          if (k > lmax) lmax = k;
        }
        // Prefetch the directory slot this entry will write to.
        __builtin_prefetch(
            &directory_[e->hash & dir_mask_], 1, 0);
        ring[(head + count) % kPFDist] = {e->hash, e};
        count++;
        if (count == kPFDist)
          drain_one();
      });
      while (count > 0)
        drain_one();

      if (key0_offset_ >= 0) {
        if (lmin > lmax) return;
        int64_t cur = key0_min_.load(std::memory_order_relaxed);
        while (lmin < cur &&
               !key0_min_.compare_exchange_weak(cur, lmin,
                                                std::memory_order_relaxed)) {}
        cur = key0_max_.load(std::memory_order_relaxed);
        while (lmax > cur &&
               !key0_max_.compare_exchange_weak(cur, lmax,
                                                std::memory_order_relaxed)) {}
      }
    };
    pool->ParallelFor(total, morsel,
                      [&](uint64_t b, uint64_t e, uint32_t) { link(b, e); });
  } else {
    zero_dir(0, dir_size);
    int64_t lmin = INT64_MAX, lmax = INT64_MIN;
    for_each_entry(0, total, [&](Entry *e) {
      if (key0_offset_ >= 0) {
        int64_t k;
        std::memcpy(&k, e->Row() + key0_offset_, sizeof(k));
        if (k < lmin) lmin = k;
        if (k > lmax) lmax = k;
      }
      auto &head = directory_[e->hash & dir_mask_];
      uintptr_t old = head.load(std::memory_order_relaxed);
      e->next = static_cast<Entry *>(bt_decode(old));
      head.store(bt_encode(e, old, e->hash), std::memory_order_relaxed);
    });
    if (key0_offset_ >= 0) {
      key0_min_.store(lmin, std::memory_order_relaxed);
      key0_max_.store(lmax, std::memory_order_relaxed);
    }
  }
  finalized_ = true;

  if (trace) {
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
    std::fprintf(stderr,
                 "[AQP-QJIT] ht_finalize n=%llu dir=%llu mode=%s %.3f ms\n",
                 (unsigned long long)n, (unsigned long long)dir_size,
                 parallel ? "parallel" : "serial", ms);
  }
}

// ---------------------------------------------------------------------------
// QjitAggState
// ---------------------------------------------------------------------------

QjitAggState::QjitAggState(std::vector<QjitAggCellDesc> descs,
                           QjitStringArena *arena)
    : descs_(std::move(descs)), cells_(descs_.size()), arena_(arena) {
  for (auto &c : cells_) {
    std::memset(&c, 0, sizeof(c));
  }
}

void QjitAggState::UpdateI64(size_t i, int64_t v) {
  QjitAggCell &c = cells_[i];
  switch (descs_[i].fn) {
  case QjitAggFn::Min:
    if (!c.seen || v < c.i64)
      c.i64 = v;
    break;
  case QjitAggFn::Max:
    if (!c.seen || v > c.i64)
      c.i64 = v;
    break;
  case QjitAggFn::Sum:
    c.i64 += v;
    break;
  case QjitAggFn::Average:
    c.i64 += v;
    c.count++;
    break;
  case QjitAggFn::Count:
  case QjitAggFn::CountStar:
    c.count++;
    break;
  }
  c.seen = true;
}

void QjitAggState::UpdateF64(size_t i, double v) {
  QjitAggCell &c = cells_[i];
  switch (descs_[i].fn) {
  case QjitAggFn::Min:
    if (!c.seen || v < c.f64)
      c.f64 = v;
    break;
  case QjitAggFn::Max:
    if (!c.seen || v > c.f64)
      c.f64 = v;
    break;
  case QjitAggFn::Sum:
    c.f64 += v;
    break;
  case QjitAggFn::Average:
    c.f64 += v;
    c.count++;
    break;
  case QjitAggFn::Count:
  case QjitAggFn::CountStar:
    c.count++;
    break;
  }
  c.seen = true;
}

void QjitAggState::UpdateStr(size_t i, const QjitString &v) {
  QjitAggCell &c = cells_[i];
  switch (descs_[i].fn) {
  case QjitAggFn::Min:
    if (!c.seen || StringCmp(v, c.str) < 0)
      c.str = arena_->Copy(v);
    break;
  case QjitAggFn::Max:
    if (!c.seen || StringCmp(v, c.str) > 0)
      c.str = arena_->Copy(v);
    break;
  case QjitAggFn::Sum:
    assert(false && "sum over strings");
    break;
  case QjitAggFn::Count:
  case QjitAggFn::CountStar:
    c.count++;
    break;
  }
  c.seen = true;
}

void QjitAggState::Merge(const QjitAggState &other) {
  assert(other.cells_.size() == cells_.size());
  for (size_t i = 0; i < cells_.size(); i++) {
    const QjitAggCell &o = other.cells_[i];
    if (!o.seen)
      continue;
    QjitAggCell &c = cells_[i];
    switch (descs_[i].fn) {
    case QjitAggFn::Min:
    case QjitAggFn::Max: {
      bool take;
      switch (descs_[i].dtype) {
      case QjitAggDType::I64:
        take = !c.seen || (descs_[i].fn == QjitAggFn::Min ? o.i64 < c.i64
                                                          : o.i64 > c.i64);
        if (take)
          c.i64 = o.i64;
        break;
      case QjitAggDType::F64:
        take = !c.seen || (descs_[i].fn == QjitAggFn::Min ? o.f64 < c.f64
                                                          : o.f64 > c.f64);
        if (take)
          c.f64 = o.f64;
        break;
      case QjitAggDType::Str:
        take = !c.seen || (descs_[i].fn == QjitAggFn::Min
                               ? StringCmp(o.str, c.str) < 0
                               : StringCmp(o.str, c.str) > 0);
        if (take)
          c.str = arena_->Copy(o.str);
        break;
      }
      break;
    }
    case QjitAggFn::Sum:
      if (descs_[i].dtype == QjitAggDType::F64)
        c.f64 += o.f64;
      else
        c.i64 += o.i64;
      break;
    case QjitAggFn::Average:
      if (descs_[i].dtype == QjitAggDType::F64)
        c.f64 += o.f64;
      else
        c.i64 += o.i64;
      c.count += o.count;
      break;
    case QjitAggFn::Count:
    case QjitAggFn::CountStar:
      c.count += o.count;
      break;
    }
    c.seen = true;
  }
}

// ---------------------------------------------------------------------------
// QjitTable
// ---------------------------------------------------------------------------

QjitTable::QjitTable(std::vector<ColumnDesc> cols, uint32_t num_workers)
    : cols_(std::move(cols)), partitions_(num_workers) {
  for (auto &p : partitions_)
    p.cols.resize(cols_.size());
}

uint32_t QjitTable::ElemSize(size_t col) const {
  switch (cols_[col].dtype) {
  case AQP_DTYPE_INT32:
    return 4;
  case AQP_DTYPE_INT64:
    return 8;
  case AQP_DTYPE_DOUBLE:
    return 8;
  case AQP_DTYPE_VARCHAR:
    return sizeof(QjitString);
  default:
    assert(false && "unsupported qjit table dtype");
    return 0;
  }
}

void QjitTable::AppendBytes(uint32_t worker, size_t col, const void *src) {
  PartCol &pc = partitions_[worker].cols[col];
  uint32_t sz = ElemSize(col);
  std::memcpy(pc.values.Allocate(sz), src, sz);
  *pc.nulls.Allocate(1) = 0;
}

void QjitTable::AppendI32(uint32_t worker, size_t col, int32_t v) {
  assert(cols_[col].dtype == AQP_DTYPE_INT32);
  AppendBytes(worker, col, &v);
}
void QjitTable::AppendI64(uint32_t worker, size_t col, int64_t v) {
  assert(cols_[col].dtype == AQP_DTYPE_INT64);
  AppendBytes(worker, col, &v);
}
void QjitTable::AppendF64(uint32_t worker, size_t col, double v) {
  assert(cols_[col].dtype == AQP_DTYPE_DOUBLE);
  AppendBytes(worker, col, &v);
}
void QjitTable::AppendStr(uint32_t worker, size_t col, const QjitString &v) {
  assert(cols_[col].dtype == AQP_DTYPE_VARCHAR);
  // Deep copy into the worker-local arena: source chunk lifetime doesn't
  // matter and concurrent workers never touch the same arena.
  QjitString owned = partitions_[worker].arena.Copy(v);
  AppendBytes(worker, col, &owned);
}
void QjitTable::AppendNull(uint32_t worker, size_t col) {
  PartCol &pc = partitions_[worker].cols[col];
  uint32_t sz = ElemSize(col);
  std::memset(pc.values.Allocate(sz), 0, sz);
  *pc.nulls.Allocate(1) = 1;
}

void QjitTable::Finalize() {
  if (finalized_)
    return;
  nrows_ = 0;
  for (const auto &p : partitions_)
    nrows_ += p.nrows;
  flat_.resize(cols_.size());
  for (size_t c = 0; c < cols_.size(); c++) {
    FlatCol &fc = flat_[c];
    uint32_t sz = ElemSize(c);
    fc.data.resize(nrows_ * sz);
    fc.validity.assign((nrows_ + 63) / 64, ~uint64_t(0));
    uint64_t row = 0;
    for (const auto &p : partitions_) {
      const PartCol &pc = p.cols[c];
      assert(pc.nulls.TotalSize() == p.nrows && "column appends != FinishRow count");
      uint64_t idx = 0;
      for (const auto &chunk : pc.values.Chunks()) {
        std::memcpy(fc.data.data() + (row + idx) * sz, chunk.data.get(),
                    chunk.used);
        idx += chunk.used / sz;
      }
      uint64_t nr = 0;
      for (const auto &nchunk : pc.nulls.Chunks()) {
        for (uint64_t b = 0; b < nchunk.used; b++) {
          if (nchunk.data[b])
            SetRowInvalid(fc.validity.data(), row + nr);
          nr++;
        }
      }
      row += p.nrows;
    }
  }
  finalized_ = true;
}

void QjitTable::ReserveFlat(uint64_t total_rows) {
  flat_.resize(cols_.size());
  for (size_t c = 0; c < cols_.size(); c++) {
    uint32_t sz = ElemSize(c);
    flat_[c].data.resize(total_rows * sz);
    flat_[c].validity.assign((total_rows + 63) / 64, ~uint64_t(0));
  }
}

int32_t QjitTable::GetI32(size_t col, uint64_t row) const {
  return reinterpret_cast<const int32_t *>(flat_[col].data.data())[row];
}
int64_t QjitTable::GetI64(size_t col, uint64_t row) const {
  return reinterpret_cast<const int64_t *>(flat_[col].data.data())[row];
}
double QjitTable::GetF64(size_t col, uint64_t row) const {
  return reinterpret_cast<const double *>(flat_[col].data.data())[row];
}
QjitString QjitTable::GetStr(size_t col, uint64_t row) const {
  return reinterpret_cast<const QjitString *>(flat_[col].data.data())[row];
}

void QjitTable::FillView(QjitTableView *view,
                         std::vector<QjitColView> *cols) const {
  assert(finalized_);
  cols->resize(cols_.size());
  for (size_t c = 0; c < cols_.size(); c++) {
    (*cols)[c].data = const_cast<uint8_t *>(flat_[c].data.data());
    (*cols)[c].validity = const_cast<uint64_t *>(flat_[c].validity.data());
    (*cols)[c].dtype = cols_[c].dtype;
    (*cols)[c].reserved = 0;
  }
  view->cols = cols->data();
  view->nrows = nrows_;
  view->ncols = cols_.size();
}

void QjitTable::BeginOutput(uint32_t worker, QjitTableColHandle *handles,
                            uint64_t ncols) {
  Partition &p = partitions_[worker];
  for (uint64_t c = 0; c < ncols; c++) {
    PartCol &pc = p.cols[c];
    uint32_t sz = ElemSize(c);
    pc.values.EnsureRoom(sz);
    pc.values.BackChunkState(&handles[c].val_cursor, &handles[c].val_limit);
    pc.nulls.EnsureRoom(1);
    pc.nulls.BackChunkState(&handles[c].null_cursor, &handles[c].null_limit);
  }
}

void QjitTable::ColSlow(uint32_t worker, size_t col,
                         QjitTableColHandle *handle, uint64_t elem_size) {
  PartCol &pc = partitions_[worker].cols[col];
  pc.values.CommitBackChunk(handle->val_cursor);
  pc.values.Allocate(elem_size);
  pc.values.BackChunkState(&handle->val_cursor, &handle->val_limit);
  handle->val_cursor -= elem_size;
}

void QjitTable::NullSlow(uint32_t worker, size_t col,
                          QjitTableColHandle *handle) {
  PartCol &pc = partitions_[worker].cols[col];
  pc.nulls.CommitBackChunk(handle->null_cursor);
  pc.nulls.Allocate(1);
  pc.nulls.BackChunkState(&handle->null_cursor, &handle->null_limit);
  handle->null_cursor -= 1;
}

void QjitTable::EndOutput(uint32_t worker, QjitTableColHandle *handles,
                           uint64_t ncols, uint64_t nrows) {
  Partition &p = partitions_[worker];
  for (uint64_t c = 0; c < ncols; c++) {
    PartCol &pc = p.cols[c];
    pc.values.CommitBackChunk(handles[c].val_cursor);
    pc.nulls.CommitBackChunk(handles[c].null_cursor);
  }
  p.nrows += nrows;
}

void QjitTable::StrCopy(uint32_t worker, QjitString *dst,
                         const QjitString *src) {
  *dst = partitions_[worker].arena.Copy(*src);
}

} // namespace qjit

// ---------------------------------------------------------------------------
// extern "C" entry points (qjit_parallel_for lives in the scheduler TU)
// ---------------------------------------------------------------------------

extern "C" {

void *qjit_buffer_grow(void *buffer, uint64_t bytes) {
  return static_cast<qjit::QjitBuffer *>(buffer)->Allocate(bytes);
}

void *qjit_ht_append(void *ht, uint32_t worker_id, uint64_t hash) {
  return static_cast<qjit::QjitHashTable *>(ht)->AppendRow(worker_id, hash);
}

void qjit_ht_begin(void *ht, uint32_t worker_id, QjitHtAppendHandle *handle) {
  static_cast<qjit::QjitHashTable *>(ht)->BeginAppend(worker_id, handle);
}

void *qjit_ht_append_slow(void *ht, uint32_t worker_id, uint64_t hash,
                           QjitHtAppendHandle *handle) {
  return static_cast<qjit::QjitHashTable *>(ht)->AppendRowSlow(
      worker_id, hash, handle);
}

void qjit_ht_end(void *ht, uint32_t worker_id, QjitHtAppendHandle *handle) {
  static_cast<qjit::QjitHashTable *>(ht)->EndAppend(worker_id, handle);
}

void qjit_ht_finalize(void *ctx, void *ht) {
  auto *qctx = static_cast<QjitQueryContext *>(ctx);
  static_cast<qjit::QjitHashTable *>(ht)->Finalize(
      static_cast<qjit::QjitWorkerPool *>(qctx->pool));
}

void *qjit_ht_dir(void *ht) {
  return const_cast<uintptr_t *>(
      static_cast<qjit::QjitHashTable *>(ht)->DirData());
}

uint64_t qjit_ht_mask(void *ht) {
  return static_cast<qjit::QjitHashTable *>(ht)->DirMask();
}

int64_t qjit_ht_key0_min(void *ht) {
  return static_cast<qjit::QjitHashTable *>(ht)->Key0Min();
}

int64_t qjit_ht_key0_max(void *ht) {
  return static_cast<qjit::QjitHashTable *>(ht)->Key0Max();
}

uint64_t qjit_ht_entries(void *ht) {
  return static_cast<qjit::QjitHashTable *>(ht)->NumEntries();
}

void qjit_agg_update_i64(void *state, uint64_t cell, int64_t v) {
  static_cast<qjit::QjitAggState *>(state)->UpdateI64(cell, v);
}

void qjit_agg_update_str(void *state, uint64_t cell, const QjitString *v) {
  static_cast<qjit::QjitAggState *>(state)->UpdateStr(cell, *v);
}

void qjit_agg_update_count(void *state, uint64_t cell) {
  static_cast<qjit::QjitAggState *>(state)->UpdateCount(cell);
}

void qjit_str_arena_copy(void *arena, QjitString *dst, const QjitString *src) {
  *dst = static_cast<qjit::QjitStringArena *>(arena)->Copy(*src);
}

void qjit_table_append_i32(void *table, uint32_t worker_id, uint64_t col,
                           int32_t v) {
  static_cast<qjit::QjitTable *>(table)->AppendI32(worker_id, col, v);
}

void qjit_table_append_i64(void *table, uint32_t worker_id, uint64_t col,
                           int64_t v) {
  static_cast<qjit::QjitTable *>(table)->AppendI64(worker_id, col, v);
}

void qjit_table_append_str(void *table, uint32_t worker_id, uint64_t col,
                           const QjitString *v) {
  static_cast<qjit::QjitTable *>(table)->AppendStr(worker_id, col, *v);
}

void qjit_table_append_null(void *table, uint32_t worker_id, uint64_t col) {
  static_cast<qjit::QjitTable *>(table)->AppendNull(worker_id, col);
}

void qjit_table_finish_row(void *table, uint32_t worker_id) {
  static_cast<qjit::QjitTable *>(table)->FinishRow(worker_id);
}

void qjit_table_begin(void *table, uint32_t worker_id,
                       QjitTableColHandle *handles, uint64_t ncols) {
  static_cast<qjit::QjitTable *>(table)->BeginOutput(worker_id, handles, ncols);
}

void qjit_table_col_slow(void *table, uint32_t worker_id, uint64_t col,
                          QjitTableColHandle *handle, uint64_t elem_size) {
  static_cast<qjit::QjitTable *>(table)->ColSlow(worker_id, col, handle,
                                                  elem_size);
}

void qjit_table_null_slow(void *table, uint32_t worker_id, uint64_t col,
                           QjitTableColHandle *handle) {
  static_cast<qjit::QjitTable *>(table)->NullSlow(worker_id, col, handle);
}

void qjit_table_end(void *table, uint32_t worker_id,
                     QjitTableColHandle *handles, uint64_t ncols,
                     uint64_t nrows) {
  static_cast<qjit::QjitTable *>(table)->EndOutput(worker_id, handles, ncols,
                                                    nrows);
}

void qjit_table_str_copy(void *table, uint32_t worker_id,
                          QjitString *dst, const QjitString *src) {
  static_cast<qjit::QjitTable *>(table)->StrCopy(worker_id, dst, src);
}

} // extern "C"
