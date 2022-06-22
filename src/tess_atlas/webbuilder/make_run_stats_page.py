"""
#TODO:
1. add to webpage
2. 'all' is not inclusive of runs that failed before logging Time
3. automate process during the building of webpages

#TODO: combine with analysis.stats_plotter
#TODO: combine with slurm_utils
"""
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tess_atlas.data.exofop import get_tic_database, filter_db_without_lk
import pandas as pd
import os
import re
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

URL = (
    "http://catalog.tess-atlas.cloud.edu.au/content/toi_notebooks/toi_{}.html"
)


def toi_num(f):
    toi_str = re.search(r"toi_(.*\d)", f).group()
    return int(toi_str.split("_")[1])


def do_tois_have_netcdf(notebook_root, all_tois):
    files = glob.glob(f"{notebook_root}/*/toi_*_files/*.netcdf")
    tois = [toi_num(f) for f in files]
    return [True if i in tois else False for i in all_tois]


def do_tois_have_phase_plot(notebook_root, all_tois):
    files = glob.glob(f"{notebook_root}/*/toi_*_files/phase*_1.png")
    tois = [toi_num(f) for f in files]
    return [True if i in tois else False for i in all_tois]


def load_run_stats(notebook_root):
    fname = glob.glob(f"{notebook_root}/*/run_stats.csv")
    run_stats = pd.read_csv(fname[0])
    cols = ["toi_numbers", "execution_complete", "duration"]
    run_stats.columns = cols
    run_stats.duration = run_stats.duration.apply(lambda x: round(x / 3600, 2))
    # keep only the longest duration for the TOI (the shorter one is just generation)
    run_stats = run_stats.sort_values(by="duration", ascending=False)
    run_stats["duplicate"] = run_stats.duplicated("toi_numbers", keep="first")
    run_stats = run_stats[run_stats["duplicate"] == False]
    return run_stats[cols]


def parse_logs(notebook_root):
    logs = glob.glob(f"{notebook_root}/log_pe/*.log")
    toi_nums = []
    log_line = []
    for l in tqdm(logs, desc="Parsing logs"):
        toi_nums.append(get_toi_number_from_log(l))
        log_line.append(get_log_last_line(l))
    df = pd.DataFrame(
        dict(toi_numbers=toi_nums, log_fn=logs, log_line=log_line)
    )
    count_before = len(df)
    df.dropna(inplace=True)
    count_after = len(df)
    if count_before != count_after:
        print(
            f"{count_before-count_after} log(s) dropped ({count_after} logs remain)"
        )
    df = df.astype({"toi_numbers": "int32"})
    return df


def get_log_last_line(log_fn):
    with open(log_fn, "rb") as f:  # open in binary mode to seek from end
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
        return last_line


def get_toi_number_from_log(log_fn):
    regex = r"run_toi\((.*\d)\)"
    with open(log_fn, "r") as f:
        f.seek(50, 0)
        txt = f.read(200)  # read the chars from idx 50 - 250
        match = re.findall(regex, txt)
        if match:
            return int(match[0])
        else:
            return np.nan


def get_toi_categories():
    tic_db = get_tic_database()
    tic_db = filter_db_without_lk(tic_db, remove=True)
    tic_db = tic_db[["TOI int", "Multiplanet System", "Single Transit"]]
    tic_db["Normal System"] = (~tic_db["Single Transit"]) & (
        ~tic_db["Multiplanet System"]
    )
    tic_db.rename(columns={"TOI int": "toi_numbers"}, inplace=True)
    return tic_db


def load_toi_summary_data(notebook_root):
    df = pd.read_csv(f"{notebook_root}/tois.csv")["toi_numbers"]
    run_stats = load_run_stats(notebook_root)
    log_df = parse_logs(notebook_root)
    categories = get_toi_categories()
    df = pd.merge(df, run_stats, how="left", on="toi_numbers")
    df = pd.merge(df, categories, how="left", on="toi_numbers")
    df = pd.merge(df, log_df, how="left", on="toi_numbers")
    df["url"] = [URL.format(i) for i in df.toi_numbers]
    df["execution_complete"] = df["execution_complete"].fillna(False)
    df["phaseplt_present"] = do_tois_have_netcdf(notebook_root, df.toi_numbers)
    df["netcdf_present"] = do_tois_have_phase_plot(
        notebook_root, df.toi_numbers
    )
    df["STATUS"] = get_status(df)
    df["TOI"] = create_gsheet_url(df)
    df["category"] = get_category(df)
    df["logs"] = format_logs(df)
    return df[["TOI", "STATUS", "category", "duration", "logs", ""]]


def get_status(df):
    status = []
    for index, toi in df.iterrows():
        s = "FAIL"
        if toi.execution_complete:
            s = "PASS"
        if not toi.phaseplt_present:
            s = "FAIL: no phaseplot"
        if not toi.netcdf_present:
            s = "FAIL: no netcdf"
        status.append(s)
    return status


def create_gsheet_url(df):
    gsheet_url = []
    for index, toi in df.iterrows():
        gsheet_url.append(f"""=hyperlink("{toi.url}",{toi.toi_numbers})""")
    return gsheet_url


def get_category(df):
    cat = []
    for index, toi in df.iterrows():
        if toi["Multiplanet System"] and toi["Single Transit"]:
            cat.append("multi planet - single transit")
        elif toi["Multiplanet System"] and not toi["Single Transit"]:
            cat.append("multi planet")
        elif not toi["Multiplanet System"] and toi["Single Transit"]:
            cat.append("single transit")
        else:
            cat.append("normal")
    return cat


def format_logs(df):
    logs = []
    for index, toi in df.iterrows():
        l = ""
        if not toi.STATUS == "PASS":
            l = f"{toi.log_fn}:{toi.log_line}"
        logs.append(l)
    return logs


def plot_runtimes(fname):
    all = pd.read_csv(fname)
    all["hours"] = all["duration_in_s"] / 3600
    s_runs = runtime_data[all["execution_complete"] == True]
    f_runs = runtime_data[all["execution_complete"] == False]
    fig, ax = plt.subplots(1, 1)
    kwgs = dict(bins=50, histtype="step", lw=3)
    ax.hist(
        runtime_data["hours"],
        color="tab:blue",
        **kwgs,
        label=f"All ({len(all)})",
    )
    ax.hist(
        s_runs["hours"],
        color="tab:green",
        **kwgs,
        label=f"Successful ({len(s_runs)})",
    )
    ax.hist(
        f_runs["hours"],
        color="tab:red",
        **kwgs,
        label=f"Failed ({len(f_runs)})",
    )
    ax.set_ylabel("# TOIs")
    ax.set_xlabel("Hours")
    ax.legend()
    fig.savefig("plot.png")


plot_runtimes("run_stats.csv")
