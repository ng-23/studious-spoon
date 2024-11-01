from torchvision.transforms import InterpolationMode
from marshmallow import Schema, fields, validate, ValidationError, pre_load, post_load, validates_schema

SCHEMA_TYPES = {'optimizer','lr_scheduler','transform','data','model','misc'}
registered_schemas = {}
registered_schemas_types = {}

def register_schema(name, typ):
    def wrapper(cls):
        registered_schemas[name] = cls
        if typ not in SCHEMA_TYPES:
            raise Exception(f'No such schema type {typ}')
        registered_schemas_types[name] = typ
        return cls
    return wrapper

@register_schema('CenterCrop', 'transform')
class CenterCropTransformSchema(Schema):
    size = fields.List(
        fields.Integer, 
        required=True,
        validate=validate.Length(equal=2),
        )
    
@register_schema('Grayscale', 'transform')
class GrayscaleTransformSchema(Schema):
    num_output_channels = fields.Integer(
        load_default=1, 
        validate=validate.OneOf([1,2,3]),
        )
    
@register_schema('RandomHorizontalFlip', 'transform')
class RandomHorizontalFlipTransformSchema(Schema):
    p = fields.Float(
        load_default=0.5, 
        validate=validate.Range(0.0, max=1.0),
        )
    
@register_schema('RandomVerticalFlip', 'transform')
class RandomVerticalFlipTransformSchema(Schema):
    p = fields.Float(
        load_default=0.5,
        validate=validate.Range(0.0, max=1.0),
        )
    
@register_schema('Resize', 'transform')
class ResizeTransformSchema(Schema):
    size = fields.List(
        fields.Integer, 
        required=True,
        validate=validate.Length(equal=2),
        )
    
    interpolation = fields.String(
        load_default=InterpolationMode.BILINEAR.value, 
        validate=validate.OneOf([member_name.lower() for member_name in InterpolationMode._member_names_]),
        )
    
    @pre_load(pass_many=False)
    def create_interps_mapping(self, data, many, **kwargs):
        # TODO: kinda an ugly hack...is there a better way?
        self.interps_mapping = {
            'nearest':InterpolationMode.NEAREST,
            'nearest_exact':InterpolationMode.NEAREST_EXACT,
            'bilinear':InterpolationMode.BILINEAR,
            'bicubic':InterpolationMode.BICUBIC,
            'box':InterpolationMode.BOX,
            'hamming':InterpolationMode.HAMMING,
            'lanczos':InterpolationMode.LANCZOS,
            }
        return data
    
    @post_load
    def convert_interpolation_mode(self, data, **kwargs):
        data['interpolation'] = self.interps_mapping[data['interpolation']]
        return data

@register_schema('ToTensor', 'transform')  
class ToTensorTransformSchema(Schema):
    # this transform is special in that it takes no arguments
    # we define a schema for it for consistency sake
    pass
    
@register_schema('Normalize', 'transform')
class NormalizeTransformSchema(Schema):
    mean = fields.List(
        fields.Float, 
        required=True,
        validate=validate.Length(min=1,max=3),
        )
    std = fields.List(
        fields.Float, 
        required=True,
        validate=validate.Length(min=1,max=3),
        )
    
    @validates_schema(pass_many=False)
    def validate_eq_lens(self, data, many, **kwargs):
        if len(data['mean']) != len(data['std']):
            raise ValidationError('Mean and standard deviation lists must have same number of channels (length)')

@register_schema('TransformsPipeline', 'transform')
class TransformsPipelineConfigSchema(Schema):
    CenterCrop = fields.Nested(CenterCropTransformSchema)
    Grayscale = fields.Nested(GrayscaleTransformSchema)
    RandomHorizontalFlip = fields.Nested(RandomHorizontalFlipTransformSchema)
    RandomVerticalFlip = fields.Nested(RandomVerticalFlipTransformSchema)
    Resize = fields.Nested(ResizeTransformSchema)
    ToTensor = fields.Nested(ToTensorTransformSchema)
    Normalize = fields.Nested(NormalizeTransformSchema)

    @pre_load(pass_many=False)
    def get_original_order(self, data, many, **kwargs):
        self.original_keys_order = [key for key in data]
        return data

    @post_load
    def reorder_original(self, data, **kwargs):
        return {key:data[key] for key in self.original_keys_order}

@register_schema('DataLoader', 'data')
class DataLoaderConfigSchema(Schema):
    batch_size = fields.Integer(load_default=1)
    shuffle = fields.Boolean(load_default=False)
    num_workers = fields.Integer(load_default=0)
    pin_memory = fields.Boolean(load_default=False)
    drop_last = fields.Boolean(load_default=False)
    timeout = fields.Integer(load_defualt=0)
    persistent_workers = fields.Boolean(load_default=False)

@register_schema('Adam', 'optimizer')
class AdamConfigSchema(Schema):
    lr = fields.Float(load_default=0.001)
    betas = fields.List(
        fields.Float, 
        load_default=[0.9,0.999], 
        validate=validate.Length(equal=2),
        )
    eps = fields.Float(load_default=1e-8)
    weight_decay = fields.Float(load_default=0.0)

@register_schema('AdamW', 'optimizer')
class AdamWConfigSchema(AdamConfigSchema):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['weight_decay'].load_default = 0.01

@register_schema('SGD', 'optimizer')
class SGDConfigSchema(Schema):
    lr = fields.Float(load_default=0.001)
    momentum = fields.Float(load_default=0.0)
    weight_decay = fields.Float(load_default=0.0)

@register_schema('RMSprop', 'optimizer')
class RMSpropConfigSchema(Schema):
    lr = fields.Float(load_default=0.01)
    alpha = fields.Float(load_default=0.99)
    eps = fields.Float(load_default=1e-8)
    weight_decay = fields.Float(load_default=0.0)
    momentum = fields.Float(load_default=0.0)

@register_schema('StepLR', 'lr_scheduler')
class StepLRConfigSchema(Schema):
    step_size = fields.Integer(required=True)
    gamma = fields.Float(load_default=0.1)
    last_epoch = fields.Integer(load_default=-1)

@register_schema('CosineAnnealingLR', 'lr_scheduler')
class CosineAnnealingLRConfigSchema(Schema):
    T_max = fields.Integer(required=True)
    eta_min = fields.Float(load_default=0.0)
    last_epoch = fields.Integer(load_default=-1)

@register_schema('ExponentialLR', 'lr_scheduler')
class ExponentialLRConfigSchema(Schema):
    gamma = fields.Float(required=True)
    last_epoch = fields.Integer(load_default=-1)

@register_schema('EarlyStopper', 'misc')
class EarlyStopperConfigSchema(Schema):
    patience = fields.Integer(load_default=1)
    min_delta = fields.Float(load_default=0.0)

@register_schema('ModelEMA', 'model')
class ModelEMAConfigSchema(Schema):
    decay = fields.Float(load_default=0.999)
    steps = fields.Integer(load_default=1)

def validate_config(config:dict, schema_name:str):
    if schema_name not in registered_schemas:
        raise Exception(f'Cannot validate supplied config, no such schema {schema_name} exists to validate against')
    schema = registered_schemas[schema_name]()

    return schema.load(config)

def validate_search_space(search_space:dict, schema_name:str):
    if schema_name not in registered_schemas:
        raise Exception(f'Cannot validate supplied search space, no such schema {schema_name} exists to validate against')
    schema = registered_schemas[schema_name]

    # search space is simply a config where each field is list of possible values
    search_space_schema = Schema.from_dict({field_name:fields.List(schema._declared_fields[field_name]) for field_name in schema._declared_fields})()

    return search_space_schema.load(search_space)
    