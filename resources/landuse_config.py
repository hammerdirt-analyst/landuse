import pandas as pd
import typing
from os import listdir
from os.path import join

def scale_the_column(x):
    xmin = x.min()
    xmax = x.max()
    xscaled = (x - xmin) / (xmax - xmin)
    
    return xscaled


def surface_area_of_feature(data, locations, columns):
    d = data[data.location.isin(locations)]
    d = d.groupby(columns, as_index=False).surface.sum()
    
    return d


def account_for_undefined(data, var_col="OBJVAL", var_label="Undefined", metric="surface", total_metric=None):
    
    data[var_col] = var_label
    data["TEMP"] = total_metric - data[metric]
    data[metric] = data["TEMP"]
    data.drop("TEMP", axis=1, inplace=True)
    
    return data


def collect_feature_data(path_to_data: str = "resources/hex-3000m"):
    # checks the files in the path_to_data are .csv
    # applies pd.DataFrame to each .csv and stores results
    # in a dictionary where key = name of map and value is
    # the corresponding dataframe
    
    files = listdir(path_to_data)
    
    data_map = {f.split('.')[0]: pd.read_csv(join(path_to_data, f)) for f in files}
    
    return data_map


def collectAggregateValues(data: pd.DataFrame = None, locations: [] = None, columns: list = ["location", "OBJVAL"],
                           to_aggregate: str = None):
    return data[data.location.isin(locations)].groupby(columns, as_index=False)[to_aggregate].sum()


def pivotValues(aggregate_values, index: str = "location", columns: str = "OBJVAL", values: str = "surface"):
    return aggregate_values.pivot(index=index, columns=columns, values=values).fillna(0)


def collectAndPivot(data: pd.DataFrame = None, locations: [] = None, columns: list = ["location", "OBJVAL"],
                    to_aggregate: str = None, scale_data: bool = False):
    # collects the geo data and aggregates the categories for a 3000 m hex
    # the total for each category is the total amount for that category in the specific 3000 m hex
    # with the center defined by the survey location.
    aggregated = collectAggregateValues(data=data, locations=locations, columns=columns, to_aggregate=to_aggregate)
    pivoted = pivotValues(aggregated, index=columns[0], columns=columns[1], values=to_aggregate)
    pivoted.columns.name = "None"
    
    if scale_data is True:
        pivoted[pivoted.columns[1:]] = pivoted[pivoted.columns[1:]].apply(lambda x: scale_the_column(x))
    
    return pivoted.reset_index(drop=False)


def define_the_objects_of_interest(data, param: str = None, param_value: typing.Union[float, int] = 0,
                                   param_label: str = None):
    # returns a list of objects whose survey value was
    # greater than the threshold value
    
    return data[data[param] > param_value][param_label].unique()


def merge_test_results_and_land_use_attributes(test_results, land_use_values, **kwargs):
    results = test_results.merge(land_use_values, **kwargs)
    return results


def test_threshhold(data, threshold, gby_column):
    # given a data frame, a threshold and a groupby column
    # the given threshold will be tested against the aggregated
    # value produced by the groupby column
    data["k"] = data.pcs_m > threshold
    exceeded = data.groupby([gby_column])['k'].sum()
    exceeded.name = "k"
    
    tested = data.groupby([gby_column]).loc_date.nunique()
    tested.name = 'n'
    
    passed = tested - exceeded
    passed.name = "n-k"
    
    ratio = exceeded / tested
    
    ratio.name = 'k/n'
    
    return exceeded, tested, passed, ratio


def test_one_object(data, threshold, gby_column):
    exceeded, tested, passed, ratio = test_threshhold(data, threshold, gby_column)
    tested = pd.concat([exceeded, tested, passed, ratio], axis=1)
    
    return tested


def group_land_use_values(collected_and_pivoted, columns, quantile, labels):
    """For groups that are considered cover and not use the magnitude of the polygon
    is taken into consideration.

    The resulting magnitudes are grouped according to the quantile variable
    """
    
    for column in columns:
        collected_and_pivoted[column] = pd.qcut(collected_and_pivoted[column], q=quantile, duplicates='drop')
    
    return collected_and_pivoted


def land_use_is_present(collected_and_pivoted, columns):
    """For groups that are superimposed or occurr infrequently only the presence
    is noted and the success rate.

    An example is a school, it takes up very little space but it does generate alot of
    activity.
    """
    
    for column in columns:
        if column in collected_and_pivoted.columns:
            collected_and_pivoted[column] = (collected_and_pivoted[column] > 0) * 1
        else:
            pass
    
    return collected_and_pivoted


def inference_for_one_attribute(data, attribute, operation):
    """The data are grouped according to attribute and aggregated
    according to operation. The results are tallied here

    The index is on the attribute range and labeled according
    to the index position. It can be accessed in two ways
    and searched by magnitude.
    """
    
    d = data.groupby(attribute).agg(operation)
    
    d["n-k"] = d["n"] - d["k"]
    d["rate"] = d["k"] / d["n"]
    d["odds"] = d["k"] / d["n-k"]
    d["pass_rate"] = d["n-k"] / d["n"]
    total_probability = d["rate"].sum()
    d["posterior"] = d["rate"] / total_probability
    d.reset_index(inplace=True, drop=False)
    d["label"] = d.index
    d.set_index(attribute, drop=True, inplace=True)
    
    return d


class LandUseValues:
    
    def __init__(self, data_map: pd.DataFrame = None, locations: list = None, region: str = None, columns: list = None,
                 id_vals: list = None, dim_oi: int = None, to_aggregate: str = None, land_use_groups: list = None):
        self.data_map = data_map
        self.locations = locations
        self.region = region
        self.columns = columns
        self.dim_oi = dim_oi
        self.to_aggregate = to_aggregate
        self.id_vals = id_vals
        self.land_use_groups = land_use_groups
        self.land_cover = None
        self.length_of = None
        self.land_use = None
        
        # return super()__init__(self)
    
    def assign_undefined(self):
        # from the data catalog:
        # https://www.swisstopo.admin.ch/fr/geodata/landscape/tlm3d.html#dokumente
        defined_land_cover = surface_area_of_feature(self.data_map, self.locations, self.columns)
        
        # print(defined_land_cover)
        
        # there are areas on the map that are not defined by a category.
        # the total surface area of all categories is subtracted from
        # the surface area of a 3000m hex = 5845672
        defined_land_cover = account_for_undefined(defined_land_cover, total_metric=self.dim_oi)
        
        # add the undefined land-use values to the to the defined ones
        land_cover = pd.concat([self.data_map, defined_land_cover])
        
        # aggregate the geo data for each location
        # the geo data for the 3000 m hexagon surrounding the survey location
        # is aggregated into the labled categories, these records get merged with
        # survey data, keyed on location
        kwargs = dict(data=land_cover, locations=self.locations, columns=self.id_vals, to_aggregate=self.to_aggregate)
        al_locations = collectAndPivot(**kwargs)
        
        
        self.land_cover = al_locations
    
    def define_total_length(self, label: str = "total"):
        kwargs = dict(data=self.data_map, locations=self.locations, columns=self.id_vals,
                      to_aggregate=self.to_aggregate)
        al_locations = collectAndPivot(**kwargs)
        al_locations[label] = al_locations[self.land_use_groups].sum(axis=1)
        
        self.length_of = al_locations[["location", label]]
    
    def define_total_surface(self, label: str = "total"):
        kwargs = dict(data=self.data_map, locations=self.locations, columns=self.id_vals,
                      to_aggregate=self.to_aggregate)
        al_locations = collectAndPivot(**kwargs)
        al_locations[label] = al_locations[self.land_use_groups].sum(axis=1)
        
        self.land_use = al_locations[["location", label]]


class TestResultsAndLandUse(LandUseValues):
    
    def __init__(self, df: pd.DataFrame = None, threshhold: typing.Union[float, int] = None, merge_column: str = None,
                 merge_method: str = None, merge_validate: str = None, groups: list = None, presence: list = None,
                 quantiles: list = None, labels: list = None, **kwargs):
        
        self.to_test = df
        self.threshhold = threshhold
        self.merge_column = merge_column
        self.merge_method = merge_method
        self.merge_validate = merge_validate
        self.groups = groups
        self.presence = presence
        self.quantiles = quantiles
        self.labels = labels
        
        super().__init__(**kwargs)
    
    def test_and_merge(self, cover: bool = True):
        
        
        tested = test_one_object(self.to_test, self.threshhold, self.merge_column)
        
        kwargs = dict(on=self.merge_column, how=self.merge_method, validate=self.merge_validate)
        
        if cover is True:
            results = merge_test_results_and_land_use_attributes(tested, self.land_cover, **kwargs)
        else:
            results = merge_test_results_and_land_use_attributes(tested, self.length_of, **kwargs)
        
        return results
    
    def make_groups_test_presence(self, cover: bool = True, label: str = "total", regional_label: str = None,
                                  label_map: pd.Series = None):
        
        results = self.test_and_merge(cover=cover)
        # print(results.head())
        
        if self.groups is not None:
            if not cover:
                self.groups = [label]
            results = group_land_use_values(results, self.groups, self.quantiles, self.labels)
        
        if self.presence is not None:
            results = land_use_is_present(results, self.presence)
        
        if regional_label is not None:
            results[regional_label] = results[self.merge_column].apply(lambda x: label_map.loc[x])
        
        return results


class InferenceGroups:
    
    def __init__(self, results: pd.DataFrame = None, column_names: list = None, operation: dict = None):
        
        self.results = results
        self.column_names = column_names
        self.operation = operation
        self.inf_groups = None

    def make_inference_for_each_attribute(self):
        
        res = {}
        
        for name in self.column_names:
            if name in self.results.columns:
                inf = inference_for_one_attribute(self.results, name, self.operation)
                res.update({name: inf})
            else:
                pass
        
        self.inf_groups = res
    
    def apply_infgroup_labels_to_results(self):
        pass


def attach_inference_group_labels_to_survey_data(fg: pd.DataFrame = None, inf_groups: dict = None,
                                                 groups_and_presence: list = None):
    newfg = fg.copy()
    
    # order the columns according to groups_and_presence
    ls = [x for x in newfg.columns if x not in groups_and_presence]
    xgt = newfg[ls]
    cols = [x for x in groups_and_presence if x in newfg.columns]
    xgl = newfg[["location", *cols]]
    newfg = xgt.merge(xgl, on="location")
    
    for label in cols:
        ifg = inf_groups[label]
        
        newfg[label] = newfg[label].apply(lambda x: ifg.loc[x, "label"])
    
    return newfg, cols


def tries_or_fails(df, columns, probability_tables, product=True, tries_or_fails="k"):
    data = df.copy()
    for x in columns:
        w = probability_tables[x]
        data[x] = data[x].apply(lambda x: w.loc[x, tries_or_fails])
    
    if product:
        data["total"] = data[columns].prod(axis=1)
    else:
        data["total"] = data[columns].sum(axis=1)
    
    return data


class LanduseConfiguration:
    
    def __init__(self, land_use_kwargs: dict = None, test_kwargs: dict = None, label: str = None,
                 assign_undefined: bool = False,
                 length: bool = False, cover: bool = False, inf_operation: dict = {"k": "sum", "n": "sum"},
                 regional_label: str = None,
                 label_map: pd.Series = None, total: bool = False):
        
        self.land_use_kwargs = land_use_kwargs
        self.test_kwargs = test_kwargs
        self.label = label
        self.label_map = label_map
        self.regional_label = regional_label
        self.assign_undefined = assign_undefined
        self.length = length
        self.cover = cover
        self.inf_operation = inf_operation
        self.total = total
        self.test_results = None
        self.grouped_data = None
        self.inf_groups = None
        self.p_tables = None
        
        super().__init__()
    
    def groups_and_presence(self):
        
        d = TestResultsAndLandUse(**self.test_kwargs, **self.land_use_kwargs)
        
        if self.length is True:
            d.define_total_length(label=self.label)
        if self.cover is True:
            d.assign_undefined()
            # print(d.land_cover)
        if self.regional_label is not None:
            dg = d.make_groups_test_presence(regional_label=self.regional_label, label_map=self.label_map)
        else:
            dg = d.make_groups_test_presence(cover=self.cover, label=self.label)
        
        self.test_results = d
        self.grouped_data = dg
    
    def inference_groups(self):
        # create the inference groups for this land use class
        # define the column names of the inference groups
        if self.total is True:
            column_names = [self.label]
        else:
            column_names = [*self.test_kwargs["groups"], *self.test_kwargs["presence"]]
        
        if self.grouped_data is None:
            self.groups_and_presence()
        
        # group the land use features by magnitude
        inf_groups = InferenceGroups(results=self.grouped_data, column_names=column_names, operation=self.inf_operation)
        inf_groups.make_inference_for_each_attribute()
        
        # attach the the inference group labels to the survey data for each feature
        labeled_groups, cols = attach_inference_group_labels_to_survey_data(self.grouped_data, inf_groups.inf_groups,
                                                                            column_names)
        labeled_groups["conf"] = list(zip(*[labeled_groups[x] for x in cols]))
        configuration_keys = labeled_groups[["location", "conf"]].set_index("location")
        
        self.inf_groups = inf_groups
        self.p_tables = inf_groups.inf_groups
        self.labeled_groups = labeled_groups
        self.configuration_keys = configuration_keys


def select_a_land_use_conf(conf, p_tables: dict = None, vals_to_drop: tuple = None):
    """Takes the land use confiuguration from a location and sums the number of trials
    and successes.

    Vals to drop gives the option to eliminate all matching land use categories that appear
    in the conf. Example if a locations has conf (0,1,2,3,4)  and another location has conf
    (2,3,4,5,6) then land-use categories 2, 3, 4 are only counted once.

    returns a tuple k=success and n=trials
    """
    
    k = 0
    n = 0
    for i, pair in enumerate(conf):
        if vals_to_drop is not None and vals_to_drop[i][1] == pair[1]:
            pass
        else:
            # print(pair)
            # print(p_tables[pair[0]])
            d = p_tables[pair[0]]
            
            e = d.loc[d.label == pair[1], ["k", "n"]].values[0]
            
            k += e[0]
            n += e[1]
    
    return k, n


def select_a_group_of_confs(conf, p_tables, drop_vals: tuple = None):
    """Takes an array of location configurations (confs) and applies
    select_a_land_use_conf to each one.

    returns a tuple k=success and n = trials
    """
    
    failed = 0
    tried = 0
    for i, pair in enumerate(conf):
        d = p_tables[pair[0]]
        attribute = pair[0]
        d_index = d.index.name
        
        if attribute != d_index:
            print(attribute, d_index)
            print("ouch")
        # collect = [x for x in d.label if x != pair[1]]
        
        e = d[["k", "n"]].sum()
        failed += e[0]
        tried += e[1]
    return failed, tried


def inferenceTableForOneLocation(name: str = None, lake: str = None, conf: tuple = None, p_tables: dict = None,
                                 tested_groups_presence: pd.DataFrame = None, tested: pd.DataFrame = None,
                                 prior: float = None, conf_names: list = None, drop_vals: tuple = None):
    # h1 the threshold was exceeded at the location given the land use values
    # likelihood of exceeding the threshold given the land use in that hex
    # print(name)
    lk, ln = select_a_land_use_conf(conf=conf, p_tables=p_tables, vals_to_drop=drop_vals)
    failed = lk / ln
    # h2 the threshold was not exceeded with that land use configuration
    passed = (ln - lk) / ln
    
    # total probability
    # the threshold was exceeded under any land use configuration
    tk = [v["k"].sum() for k, v in p_tables.items()]
    tk = sum(tk)
    
    # the number of tries
    tn = [v["n"].sum() for k, v in p_tables.items()]
    tn = sum(tn)
    
    
    # the threshold was not exceeded
    passed_t = (tn - tk) / tn
    failed_t = tk / tn
    
    if prior is not None:
        p = prior
    else:
        # assign a prior from the survey results
        # if there is data for the location in question
        # use it.
        pkn, pnn = tested.loc[tested.location == name, ['k', 'n']].sum().values
        
        if pkn == 0:
            p = (pkn + 1) / (pnn + 2)
        elif pkn / pnn == 1:
            p = (pkn + 1) / (pnn + 2)
        else:
            p = pkn/pnn
    passed_prior = (1 - p)*passed
    failed_prior = p*failed
    inf = failed_prior / (passed_prior + failed_prior)
    
    
    
    return inf


def inference_for_one_location(location: str = None, lake: str = None, fgl_conf_keys: pd.DataFrame = None,
                               conf_columns: list = None, p_tables: dict = None, prior: float = None,
                               tested: pd.DataFrame = None, drop_vals: tuple = None, conf_label: str = None):
    conf = fgl_conf_keys.loc[location, conf_label]
    conf = tuple(zip(conf_columns, conf))
    
    p = inferenceTableForOneLocation(
        name=location,
        lake=lake,
        conf_names=conf_columns,
        conf=conf,
        p_tables=p_tables,
        tested=tested,
        prior=prior,
        drop_vals=drop_vals
    )
    
    return p
