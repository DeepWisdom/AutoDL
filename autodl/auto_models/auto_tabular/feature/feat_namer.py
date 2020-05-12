from ...auto_tabular import CONSTANT


class FeatNamer:
    @staticmethod
    def gen_feat_name(cls_name, feat_name, param, feat_type):
        prefix = CONSTANT.type2prefix[feat_type]
        if param == None:
            return "{}{}:{}".format(prefix, cls_name, feat_name)
        else:
            return "{}{}:{}:{}".format(prefix, cls_name, feat_name, param)
