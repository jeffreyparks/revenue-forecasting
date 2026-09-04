
def clean_df(df):

    import re

    def clean_name(name):
        name = name.lower().replace(' ', '_')
        name = re.sub(r'[^a-z0-9_]', '', name)
        name = name.rstrip("_,.")
        return name
    
    df.columns = [clean_name(col) for col in df.columns]
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # whitespace
    return df

def inspect_pipeline(data, shape=True, save=False, print=False, filename=None, **kwargs):
    """ Inspect data during the pipeline run, insert after transformers as needed """
    if shape:
        print('shape:',data.shape)
    if print:
        print(data)
    if save:
        dump(data, f"../data/output/inspect_{filename}.joblib")
    return(data)
    # ('inspector_1', FunctionTransformer(inspect_data, validate=False)),