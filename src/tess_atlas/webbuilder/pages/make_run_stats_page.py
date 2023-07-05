"""
#TODO:
1. add to webpage
2. 'all' is not inclusive of runs that failed before logging Time
3. automate process during the building of webpages

#TODO: combine with analysis.stats_plotter
#TODO: combine with slurm_utils
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from tess_atlas.data.planet_candidate import CLASS_SHORTHAND
from tess_atlas.plotting.runtime_plotter import plot_runtimes
from tess_atlas.utils import grep_toi_number

from .templates import IMAGE, TOI_LINK, render_page_template

URL_BASE = "http://catalog.tess-atlas.cloud.edu.au/content/toi_notebooks"
NOTEBOOK_URL = URL_BASE + "/toi_{}.html"
IMG_URL = URL_BASE + "/toi_{}_files/phase_plot_TOI{}_1_thumbnail.png"
PHASE_IMG = "phase_plot_TOI{}_1_thumbnail.png"
LINK = """<a href="{url}">{txt}</a>"""
NETCDF_REGEX = "*/toi_*_files/*.netcdf"
PHASE_REGEX = "*/toi_*_files/phase*_1_lowres.png"


def do_tois_have_netcdf(notebook_root, all_tois):
    files = glob.glob(f"{notebook_root}/{NETCDF_REGEX}")
    tois = [grep_toi_number(f) for f in files]
    return [True if i in tois else False for i in all_tois]


def do_tois_have_phase_plot(notebook_root, all_tois):
    files = glob.glob(f"{notebook_root}/{PHASE_REGEX}")
    tois = [grep_toi_number(f) for f in files]
    return [True if i in tois else False for i in all_tois]


def load_run_stats(notebook_root):
    fname = glob.glob(f"{notebook_root}/*/run_stats.csv")
    run_stats = pd.read_csv(fname[0])
    cols = ["toi_numbers", "execution_complete", "duration"]
    run_stats.columns = cols
    run_stats.duration = run_stats.duration.apply(
        lambda x: round(x / (60 * 60), 2)
    )
    # keep only the longest duration for the TOI (the shorter one is just generation)
    run_stats = run_stats.sort_values(by="duration", ascending=False)
    run_stats["duplicate"] = run_stats.duplicated("toi_numbers", keep="first")
    run_stats = run_stats[run_stats["duplicate"] == False]
    return run_stats[cols]


def parse_logs(notebook_root):
    logs = glob.glob(f"{notebook_root}/log_pe/pe_*.log")
    toi_nums = []
    log_line = []
    for l in tqdm(logs, desc="Parsing logs"):
        toi_nums.append(get_toi_number_from_log(l))
        log_line.append(get_log_last_line(l))
    df = pd.DataFrame(
        dict(toi_numbers=toi_nums, log_fn=logs, log_line=log_line)
    )
    count_before = len(df)
    df = df.dropna()
    df = df.drop_duplicates(subset="toi_numbers", keep="first")
    count_after = len(df)
    if count_before != count_after:
        print(
            f"{count_before-count_after} log(s) dropped ({count_after} logs remain)"
        )
    df = df.astype({"toi_numbers": "int32"})
    return df


def clean_log_line(log):
    log = log.strip()
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    log = ansi_escape.sub("", log)
    return log


def get_log_last_line(log_fn):
    with open(log_fn, "rb") as f:  # open in binary mode to seek from end
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
        return clean_log_line(last_line)


def get_toi_number_from_log(log_fn):
    with open(log_fn, "r") as f:
        f.seek(50, 0)
        txt = f.read(200)  # read the chars from idx 50 - 250
        toi_int = grep_toi_number(txt)
        if toi_int:
            return toi_int
        else:
            return np.nan


def get_classification(short_classes):
    return [CLASS_SHORTHAND.get(sc, sc) for sc in short_classes]


def get_exofop_data():
    tic_db = get_tic_database()
    tic_db = filter_db_without_lk(tic_db, remove=True).copy()
    tic_db["Normal System"] = (~tic_db["Single Transit"]) & (
        ~tic_db["Multiplanet System"]
    )
    tic_db.rename(columns={"TOI int": "toi_numbers"}, inplace=True)
    return tic_db


def load_toi_summary_data(notebook_root):
    df = pd.read_csv(f"{notebook_root}/tois.csv")["toi_numbers"]
    run_stats = load_run_stats(notebook_root)
    log_df = parse_logs(notebook_root)
    exofop = get_exofop_data()
    df = pd.merge(df, run_stats, how="left", on="toi_numbers")
    df = pd.merge(df, exofop, how="left", on="toi_numbers")
    df = pd.merge(df, log_df, how="left", on="toi_numbers")
    df["url"] = [URL.format(i) for i in df.toi_numbers]
    df["execution_complete"] = df["execution_complete"].fillna(False)
    df["duration"] = df["duration"].fillna(10)
    df["phaseplt_present"] = do_tois_have_phase_plot(
        notebook_root, df.toi_numbers
    )
    df["netcdf_present"] = do_tois_have_netcdf(notebook_root, df.toi_numbers)
    df["STATUS"] = get_status(df)
    df["TOI"] = create_weburl(df)
    df["Category"] = get_category(df)
    df["Logs"] = format_logs(df)
    df["Phase Plot"] = create_phase_plot_urls(df)
    df["Classification"] = get_classification(df["TESS Disposition"])
    df = df.drop_duplicates(subset="toi_numbers", keep="first")
    return df


def get_status(df):
    status = []
    for index, toi in df.iterrows():
        s = "FAIL: no netcdf"
        if toi.netcdf_present:
            s = "PASS"
        if not toi.phaseplt_present:
            s = "FAIL: no phaseplot"
        status.append(s)
    return status


def create_weburl(df):
    url = [""] * len(df)
    for index, toi in df.iterrows():
        url[index] = LINK.format(url=toi.url, txt=toi.toi_numbers)
    return url


def create_phase_plot_urls(df):
    html = """<img src="{}" alt="{}" width="80px" >"""
    urls = [""] * len(df)
    for i, toi in df.iterrows():
        t = toi.toi_numbers
        img = IMG_URL.format(t, PHASE_IMG.format(t))
        txt = html.format(img, f"TOI{t} Phaseplot")
        urls[i] = LINK.format(url=toi.url, txt=txt)
    return urls


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


def generate_table_html(dataframe: pd.DataFrame):
    dataframe["Duration[Hr]"] = dataframe["duration"]
    dataframe = dataframe[
        [
            "TOI",
            "STATUS",
            "Classification",
            "Category",
            "Duration[Hr]",
            "Phase Plot",
            "Logs",
        ]
    ]
    table_html = dataframe.to_html(table_id="table", index=False)
    table_html = table_html.replace(
        "dataframe", "table table-striped table-bordered"
    )
    table_html = table_html.replace("&lt;", "<")
    table_html = table_html.replace("&gt;", ">")
    return


def make_stats_page(notebook_root, path_to_stats_page):
    """
    notebook_root is the top level dir name of the notebooks

    eg:

    <root>
    |--log_gen/
    |--log_pe/
    |--log_web/
    |--submit/
    |--<notebook_dir>
    """
    toi_df = load_toi_summary_data(notebook_root)
    table_html = generate_table_html(toi_df)
    stats_image_path = path_to_stats_page.replace(".html", ".png")
    plot_runtimes(toi_df, savepath=stats_image_path)
    page_data = dict(table_html=table_html, stats_image_path=stats_image_path)
    page_contents = render_page_template(path_to_stats_page, page_data)

    with open(path_to_stats_page, "w") as f:
        f.write(page_contents)
