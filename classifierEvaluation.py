import pandas as pd

import re
import sys


def read_output_file(filename):
    output = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            line = tuple(line.split(","))
            output.append(line)

    df = pd.DataFrame(output)
    df.rename({0: "doc_id", 1: "author"}, axis=1, inplace=True)
    
    return df


def read_target_file(filename):
    output = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            line = line.split(",")
            line = (re.sub(r"[^0-9]", "", line[0]), line[1])
            output.append(line)

    df = pd.DataFrame(output)
    df.rename({0: "doc_id", 1: "author"}, axis=1, inplace=True)
    
    return df


def evalulate(output_file, target_file="doc_and_author.csv"):
    model_output = read_output_file(output_file)
    target = read_target_file(target_file)

    df = pd.merge(model_output, target, on="doc_id", how="inner", suffixes=("_pred", "_target"))
    authors = df["author_target"].unique()

    for author in authors:
        print("===================")
        print(author)

        target_df = df[df["author_target"] == author]
        false_positive_df = df[df["author_pred"] == author]

        true_pos = (target_df["author_pred"] == target_df["author_target"]).sum()
        false_pos = (false_positive_df["author_pred"] != false_positive_df["author_target"]).sum()
        false_neg = (target_df["author_pred"] != target_df["author_target"]).sum()

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        print("Hits:", true_pos)
        print("Strikes:", false_pos)
        print("Misses:", false_neg)
        print("Precision: %.3f" % precision)
        print("Recall: %.3f" % recall)
        print("F1: %.3f" % ((2 * precision * recall) / (precision + recall)))
        print()

    correct = (df["author_pred"] == df["author_target"]).sum()
    incorrect = (df["author_pred"] != df["author_target"]).sum()
    print("Total Correct:", correct)
    print("Total Incorrect:", incorrect)
    print("Overall Accuracy: %.3f" % (correct / (correct + incorrect)))

    confusion_matrix = pd.crosstab(df['author_target'], df['author_pred'], rownames=['Actual'], colnames=['Predicted'])
    with open(output_file + "_confusion_matrix.tsv", "w") as file:
        file.write(confusion_matrix.to_csv(sep="\t"))
        print()
        print("Confusion matrix successfully written to: %s" % (output_file + "_confusion_matrix.tsv"))
    print(confusion_matrix)


if __name__ == "__main__":
    target_filename = None
    if len(sys.argv) == 2:
        pred_filename = sys.argv[1]
    elif len(sys.argv) == 3:
        pred_filename = sys.argv[1]
        target_filename = sys.argv[2]
    else:
        raise NotImplementedError

    if target_filename is not None:
        evalulate(pred_filename, target_filename)
    else:
        evalulate(pred_filename)
