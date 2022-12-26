use std::{fs::File, io::Write};

use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::Array2;
use linfa::prelude::*;
use ndarray::prelude::*;

fn main() {
    let happy_data:Array2<f32> = array!(
        [1.,1.,1000., 1.,10.],
        [2.,1.,0., 1.,5.],
        [1.,2.,0., 1.,6.],
        [1.,1.,500., 1.,7.],
        [1.,2.,1000., 2.,8.],
        [2.,2.,800., 1.,3.],
        [2.,1.,1000., 2.,8.],
        [1.,2.,1000., 2.,8.],
        [2.,1.,0., 1.,3.],
        [2.,2.,400., 2.,4.],
        [1.,1.,200., 0.,3.],
        [1.,1.,1000., 0.,2.],
        [1.,1.,50., 1.,3.]
    );

    let feature_names = vec!["Played Video Games", "Speak to Loved one", "Worked Out", "Eat Comfort Food"];
    let num_feat = happy_data.len_of(Axis(1)) - 1;
    let features = happy_data.slice(s![.., 0..num_feat]).to_owned();
    // just happines column
    let labels = happy_data.column(num_feat).to_owned();

    // decorate the data set
    let linfa_dataset = Dataset::new(features, labels).map_targets(|x| match x.to_owned() as i32 {
        i32::MIN..=4 => "Sad",
        5..=7 => "Ok",
        8..=i32::MAX => "Joyous"

    }).with_feature_names(feature_names);

    let model = DecisionTree::params().split_quality(SplitQuality::Gini).fit(&linfa_dataset).unwrap();

    File::create("dt.tex").unwrap().write_all(model.export_to_tikz().with_legend().to_string().as_bytes());

}
