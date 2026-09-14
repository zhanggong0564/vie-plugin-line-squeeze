"""场景响应文档：合成示例，保留实际业务字段和判定语义。"""

from schemas.data_base import DetectionItem, MoMResult


RESPONSE_NOTES = (
    '### 场景明细与判定规则\n\n明细 scene=dc 或 fu，分别表示 DC 线标和 FU 线标；实际检查组和数量由 product_model 决定'
    '。明细按线序位置展开，status/verdict 为该位置的校验结论，整体按型号启用的检查组汇总。accuracy 是线标区域检测置信度，不是 OCR'
    ' 识别置信度；当前公共响应 name 为空且 ocr_tokens 为 null，不对外返回中间识别文字。非空识别组的数量不等于型号要求时，当前会直接返'
    '回 FAIL 和空明细；缺线也可能产生 coordinate=[]、accuracy=0 的 FAIL 占位项；未知产品型号为执行错误 code=100'
    '3、verdict=null。示例为五路有熔丝盒有磁环，检查 5 个 DC 和 5 个 FU 位置。\n\n本场景默认 coordinate 为原图宽高归一'
    '化的四边形八个数：[x1,y1,x2,y2,x3,y3,x4,y4]，矩形按左上、右上、右下、左下排列；x 乘原图宽、y 乘原图高可还原像素。缺失项可以'
    '为 []，不得当作原点检测框。color 是显示颜色，不作为判定依据；以 verdict 为准。vis_image 为可选 JPEG data URI，'
    '关闭可视化或绘制失败时可为空；示例使用空字符串，不填伪造 base64。 当前这些示例场景的公共 ocr_tokens 为 null；不要把内部 OCR'
    ' 结构当作既有接口字段。'
)


def _result(passed):
    items = []
    for scene, y in [("dc", 0.2), ("fu", 0.6)]:
        for index in range(5):
            x = 0.1 + index * 0.15
            items.append(DetectionItem(
                status=passed or scene != "dc" or index != 2, scene=scene, accuracy=0.96,
                coordinate=[x, y, x + 0.08, y, x + 0.08, y + 0.1, x, y + 0.1],
            ))
    for item in items:
        item.coordinate = [round(value, 4) for value in item.coordinate]
    return MoMResult(status=passed, message="检测成功" if passed else "检测失败", detailList=items).to_dict()


RESPONSE_EXAMPLES = {
    "PASS": {"summary": "通过：DC/FU 各五个位置线序正确", "result": _result(True)},
    "FAIL": {"summary": "不通过：第三个 DC 位置线序不符", "result": _result(False)},
}
