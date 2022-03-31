from datetime import datetime
import hydralit_components as hc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
import io
from st_aggrid import AgGrid
import os
import platform
import os.path, time




# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Estadia Dashboard", page_icon=":bar_chart:", layout="wide")

def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1


# -------------------------Hidralit theme-------------------------

# can apply customisation to almost all the properties of the card, including the progress bar
theme_bad = {'bgcolor': '#B22222', 'title_color': 'white', 'content_color': 'whithe', 'icon_color': 'white',
             'progress_color': 'red'}

# -------------------------- LOAD ASSETS ---------------------------
lottie_car = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_v92o72md.json")
lottie_chart = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_S2eIOQ.json")
lottie_tower = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_jznvojh3.json")
lottie_time = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_a3g6x26d.json")
lottie_truck = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_hutlkiuf.json")
lottie_list = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_emujvwjt.json")
img_control_Tower = Image.open("images/ControlTowerwhite.png")

# ---- --------------------READ EXCEL ----------------------------

# df = pd.concat(pd.read_excel(hoja_de_calculo, sheet_name=None), ignore_index=True)
# url = "https://github.com/AlanGGutierrez/GMDashEstadia/blob/main/BaseDescargaFin.xlsx"
# download = requests.get(url).content
# df = pd.concat(pd.read_excel(io.StringIO(download.decode('utf-8'))))
df = pd.concat(pd.read_excel(
    io="BaseDescargaFin.xlsx",
    engine="openpyxl",
    sheet_name=None,
    skiprows=0,
    nrows=10000,
    na_values=[" "]
), ignore_index=True)
# Dataframe General
today = datetime.now()
df["fecha_actual"] = today
# df["fecha_dia"] = (df["CARGA_PROGRAMADA"].dt.strftime('%d-%m-%Y'))
df["ARRIBO"].fillna(df["fecha_actual"], inplace=True)
df.rename(columns={'ZONA VENTA': 'ZONA_VENTA'}, inplace=True)
df["CITA DE DESCARGA"].fillna(df["fecha_actual"], inplace=True)
df['estadia_vs_arribo_sum'] = np.where(df['ESTATUS MONITOREO'] == "ENTREGADO", "0",
                                       (np.where(df['ESTATUS MONITOREO'] == "T.IDA", "0",
                                                 round((today - df["ARRIBO"]) / pd.Timedelta(hours=1),
                                                       0)))).astype("float")

# df['estadia_vs_cita'] = round((today - df["CITA DE DESCARGA REAL"]) / pd.Timedelta(hours=1), 0)
df['estadia_vs_cita_sum'] = np.where(df['ESTATUS MONITOREO'] == "ENTREGADO", "0",
                                     (np.where(df['ESTATUS MONITOREO'] == "T.IDA", "0",
                                               round((today - df["CITA DE DESCARGA"]) / pd.Timedelta(hours=1),
                                                     0)))).astype("float")
df['estadia_vs_cita_sum'] = np.where(df['estadia_vs_cita_sum'] < 0, 0, df['estadia_vs_cita_sum'])
df['estadia_vs_arribo_acum'] = np.where(df["estadia_vs_arribo_sum"] > 1, df["estadia_vs_arribo_sum"], 0).astype("float")

df.dropna(subset=["ZONA_VENTA"], inplace=True)
df = df.drop(df[df['ESTATUS MONITOREO'] == "CANCELADO"].index)
df = df.drop(df[df['ESTATUS MONITOREO'] == "CANCELADO "].index)
df = df.drop(df[df['ESTATUS MONITOREO'] == "REPROGRAMAR"].index)

csv = convert_df(df)
# AgGrid(df)
ultModi = "Ultima Actualización: %s" % time.ctime(os.path.getmtime("BaseDescargaFin.xlsx"))
#print("created: %s" % time.ctime(os.path.getctime("BaseDescargaFin.xlsx")))

# ---------------------------Dataframe top 5 de estadia----------------------------
new_df = df.filter(
    ["DESTINO", "ZONA_VENTA", "estadia_vs_arribo_sum", "estadia_vs_cita_sum", "ESTATUS MONITOREO", "TC REAL"])
new_df_sum_estadia = new_df.groupby('DESTINO')['estadia_vs_arribo_sum'].agg(['sum', "mean"]).round(2)
new_df_top_estadia = new_df_sum_estadia.sort_values('sum', ascending=False)


options = ['PEND.DESCARGA', 'DESCARGANDO']
df_pendesc_desc = df.loc[df['ESTATUS MONITOREO'].isin(options)]
#AgGrid(df_pendesc_desc)
dfpub = df.loc[df['estadia_vs_arribo_sum'] > 0]
topdfnew = dfpub.groupby('DESTINO')['estadia_vs_arribo_sum'].agg(['sum', "mean"]).round(2)
topdfnew = topdfnew.sort_values('mean', ascending=False)
topdfnew = topdfnew.reset_index()
tamdf = (len(topdfnew))
for i in range(5):
    if tamdf < 5:
        topdfnew.loc[tamdf] = ['Sin datos', 0, 0]
        tamdf = tamdf + 1


#st.dataframe(topdfnew)


# ---------------------Dataframe datos adicionales------------------------
new_df_top_estadia_top = new_df_top_estadia.reset_index()
df_top = new_df.groupby(by=["DESTINO", "ESTATUS MONITOREO"]).size().to_frame('size').reset_index()
#st.dataframe(new_df_top_estadia_top)
#st.dataframe(df_top)


# ------------------Top 1--------------------------------------
df_top1 = topdfnew.iloc[0]["DESTINO"]

df_top1_pend = np.where((df_top["DESTINO"] == df_top1) & (df_top["ESTATUS MONITOREO"] == "PEND.DESCARGA"),
                        df_top["size"], ["0"])

b1 = df_top1_pend > "0"
out1 = np.extract(b1, df_top1_pend)
if len(out1) == 0:
    out1 = "0"

df_top1_pend_out = float(listToString(out1))

df_top1_desc = np.where((df_top["DESTINO"] == df_top1) & (df_top["ESTATUS MONITOREO"] == "DESCARGANDO"), df_top["size"],
                        ["0"])
b2 = df_top1_desc > "0"
out2 = np.extract(b2, df_top1_desc)
if len(out2) == 0:
    out2 = "0"

df_top1_desc_out = float(listToString(out2))

df_top1_tida = np.where((df_top["DESTINO"] == df_top1) & (df_top["ESTATUS MONITOREO"] == "T.IDA"), df_top["size"],
                        ["0"])
b3 = df_top1_tida > "0"
out3 = np.extract(b3, df_top1_tida)
if len(out3) == 0:
    out3 = "0"

df_top1_tida_out = float(listToString(out3))

sumT1 = np.where((topdfnew["DESTINO"] == df_top1), topdfnew["sum"], ["0"])
bT1 = sumT1 > "0"
outT1 = np.extract(bT1, sumT1)
if len(outT1) == 0:
    outT1 = "0"

df_top1_sumT1_out = float(listToString(outT1))

countT1 = float(df_top1_desc_out + df_top1_pend_out)
if countT1 == 0:
    promedioT1 = 0
else:
    promedioT1 = round(df_top1_sumT1_out / countT1, 2)

# print(dfpub.dtypes)
placasT1 = df_pendesc_desc.loc[df_pendesc_desc["DESTINO"] == df_top1, ["TC REAL"]]
# placas.reset_index()
#st.dataframe(dfpub)
# print(df_top1)
# print(placasT1["TC REAL"])
#placas = np.where((dfpub["estadia_vs_arribo_sum"]>0) & (df_top["DESTINO"] == df_top1), dfpub["TC REAL"],0)
#print(placas)

# --------------------------- Top 2---------------------------------
df_top2 = topdfnew.iloc[1]["DESTINO"]

df_top2_pend = np.where((df_top["DESTINO"] == df_top2) & (df_top["ESTATUS MONITOREO"] == "PEND.DESCARGA"),
                        df_top["size"], ["0"])
b12 = df_top2_pend > "0"
out12 = np.extract(b12, df_top2_pend)
if len(out12) == 0:
    out12 = "0"

df_top2_pend_out = float(listToString(out12))

df_top2_desc = np.where((df_top["DESTINO"] == df_top2) & (df_top["ESTATUS MONITOREO"] == "DESCARGANDO"), df_top["size"],
                        ["0"])
b22 = df_top2_desc > "0"
out22 = np.extract(b22, df_top2_desc)
if len(out22) == 0:
    out22 = "0"

df_top2_desc_out = float(listToString(out22))

df_top2_tida = np.where((df_top["DESTINO"] == df_top2) & (df_top["ESTATUS MONITOREO"] == "T.IDA"), df_top["size"],
                        ["0"])
b32 = df_top2_tida > "0"
out32 = np.extract(b32, df_top2_tida)
if len(out32) == 0:
    out32 = "0"

df_top2_tida_out = float(listToString(out32))

sumT2 = np.where((topdfnew["DESTINO"] == df_top2), topdfnew["sum"], ["0"])
bT2 = sumT2 > "0"
outT2 = np.extract(bT2, sumT2)
if len(outT2) == 0:
    outT2 = "0"

df_top2_sumT2_out = float(listToString(outT2))

countT2 = float(df_top2_desc_out + df_top2_pend_out)
if countT2 == 0:
    promedioT2 = 0
else:
    promedioT2 = round(df_top2_sumT2_out / countT2, 2)

placasT2 = df_pendesc_desc.loc[df_pendesc_desc["DESTINO"] == df_top2, ["TC REAL"]]
# print(df_top2)
# print(placasT2["TC REAL"])
# ------------------------Top 3--------------------------------
df_top3 = topdfnew.iloc[2]["DESTINO"]

df_top3_pend = np.where((df_top["DESTINO"] == df_top3) & (df_top["ESTATUS MONITOREO"] == "PEND.DESCARGA"),
                        df_top["size"], ["0"])
b13 = df_top3_pend > "0"
out13 = np.extract(b13, df_top3_pend)
if len(out13) == 0:
    out13 = "0"

df_top3_pend_out = float(listToString(out13))

df_top3_desc = np.where((df_top["DESTINO"] == df_top3) & (df_top["ESTATUS MONITOREO"] == "DESCARGANDO"), df_top["size"],
                        ["0"])
b23 = df_top3_desc > "0"
out23 = np.extract(b23, df_top3_desc)
if len(out23) == 0:
    out23 = "0"

df_top3_desc_out = float(listToString(out23))

df_top3_tida = np.where((df_top["DESTINO"] == df_top3) & (df_top["ESTATUS MONITOREO"] == "T.IDA"), df_top["size"],
                        ["0"])
b33 = df_top3_tida > "0"
out33 = np.extract(b33, df_top3_tida)
if len(out33) == 0:
    out33 = "0"

df_top3_tida_out = float(listToString(out33))

sumT3 = np.where((topdfnew["DESTINO"] == df_top3), topdfnew["sum"], ["0"])
bT3 = sumT3 > "0"
outT3 = np.extract(bT3, sumT3)
if len(outT3) == 0:
    outT3 = "0"

df_top3_sumT3_out = float(listToString(outT3))

countT3 = float(df_top3_desc_out + df_top3_pend_out)
if countT3 == 0:
    promedioT3 = 0
else:
    promedioT3 = round(df_top3_sumT3_out / countT3, 2)

placasT3 = df_pendesc_desc.loc[df_pendesc_desc["DESTINO"] == df_top3, ["TC REAL"]]
print(df_top3)
print(placasT3["TC REAL"])
# ----------------------------Top 4--------------------------
df_top4 = topdfnew.iloc[3]["DESTINO"]

df_top4_pend = np.where((df_top["DESTINO"] == df_top4) & (df_top["ESTATUS MONITOREO"] == "PEND.DESCARGA"),
                        df_top["size"], ["0"])
b14 = df_top4_pend > "0"
out14 = np.extract(b14, df_top4_pend)
if len(out14) == 0:
    out14 = "0"

df_top4_pend_out = float(listToString(out14))

df_top4_desc = np.where((df_top["DESTINO"] == df_top4) & (df_top["ESTATUS MONITOREO"] == "DESCARGANDO"), df_top["size"],
                        ["0"])
b24 = df_top4_desc > "0"
out24 = np.extract(b24, df_top4_desc)
if len(out24) == 0:
    out24 = "0"

df_top4_desc_out = float(listToString(out24))

df_top4_tida = np.where((df_top["DESTINO"] == df_top4) & (df_top["ESTATUS MONITOREO"] == "T.IDA"), df_top["size"],
                        ["0"])
b34 = df_top4_tida > "0"
out34 = np.extract(b34, df_top4_tida)
if len(out34) == 0:
    out34 = "0"

df_top4_tida_out = float(listToString(out34))

sumT4 = np.where((topdfnew["DESTINO"] == df_top4), topdfnew["sum"], ["0"])
bT4 = sumT4 > "0"
outT4 = np.extract(bT4, sumT4)
if len(outT4) == 0:
    outT4 = "0"

df_top4_sumT4_out = float(listToString(outT4))

countT4 = float(df_top4_desc_out + df_top4_pend_out)
if countT4 == 0:
    promedioT4 = 0
else:
    promedioT4 = round(df_top4_sumT4_out / countT4, 2)

placasT4 = df_pendesc_desc.loc[df_pendesc_desc["DESTINO"] == df_top4, ["TC REAL"]]
# print(df_top4)
# print(placasT4["TC REAL"])
# ----------------------------Top 5------------------------------
df_top5 = topdfnew.iloc[4]["DESTINO"]

df_top5_pend = np.where((df_top["DESTINO"] == df_top5) & (df_top["ESTATUS MONITOREO"] == "PEND.DESCARGA"),
                        df_top["size"], ["0"])
b15 = df_top5_pend > "0"
out15 = np.extract(b15, df_top5_pend)
if len(out15) == 0:
    out15 = "0"

df_top5_pend_out = float(listToString(out15))

df_top5_desc = np.where((df_top["DESTINO"] == df_top5) & (df_top["ESTATUS MONITOREO"] == "DESCARGANDO"), df_top["size"],
                        ["0"])
b25 = df_top5_desc > "0"
out25 = np.extract(b25, df_top5_desc)
if len(out25) == 0:
    out25 = "0"

df_top5_desc_out = float(listToString(out25))

df_top5_tida = np.where((df_top["DESTINO"] == df_top5) & (df_top["ESTATUS MONITOREO"] == "T.IDA"), df_top["size"],
                        ["0"])
b35 = df_top5_tida > "0"
out35 = np.extract(b35, df_top5_tida)
if len(out35) == 0:
    out35 = "0"

df_top5_tida_out = float(listToString(out35))

sumT5 = np.where((topdfnew["DESTINO"] == df_top5), topdfnew["sum"], ["0"])
bT5 = sumT5 > "0"
outT5 = np.extract(bT5, sumT5)
if len(outT5) == 0:
    outT5 = "0"

df_top5_sumT5_out = float(listToString(outT5))

countT5 = float(df_top5_desc_out + df_top5_pend_out)
if countT5 == 0:
    promedioT5 = 0
else:
    promedioT5 = round(df_top5_sumT5_out / countT5, 2)

placasT5 = df_pendesc_desc.loc[df_pendesc_desc["DESTINO"] == df_top5, ["TC REAL"]]
# print(df_top5)
# print(placasT5["TC REAL"])
# ---- -----------------------------SIDEBAR ---------------------------------
st.sidebar.header("Filtra aquí:")

destino = st.sidebar.multiselect(
    "Selecciona el destino:",
    options=df["DESTINO"].unique(),
    default=df["DESTINO"].unique()
)

zona = st.sidebar.multiselect(
    "Selecciona la zona:",
    options=df["ZONA_VENTA"].unique(),
    default=df["ZONA_VENTA"].unique()
)

# fi = st.sidebar.date_input('Fecha Inicial:', key="fecha_inicial")
# ff = st.sidebar.date_input('Fecha Final:', key="fecha_final")

df_selection = df.query(
    # "ESTATUS MONITOREO == @estatus & Cerveceria == @cerveceria & Transportista == @transportista & Destino ==
    # @destino & ZONA_VENTA == @zona"
    "DESTINO == @destino & ZONA_VENTA == @zona  "
)
# st.dataframe(df_selection)

# -------------------------- MAINPAGE ----

left_column, mid_column, right_column = st.columns(3)
with left_column:
    st_lottie(lottie_car, height=300, key="car")
with mid_column:
    st.markdown("<h1 style='text-align: center; color: white;'>Estadias Dashboard</h1>", unsafe_allow_html=True)
    st.image(img_control_Tower)

with right_column:
    st_lottie(lottie_chart, height=300, key="chart")

st.markdown("""---""")

# --------------------------------KPI---------
sum_EA = df_selection["estadia_vs_arribo_sum"].sum()
countEA = (df_selection['estadia_vs_arribo_sum'] > 0).sum()
if countEA == 0:
    estadiavsArribo = 0
else:
    estadiavsArribo = round(sum_EA / countEA, 1)
sum_EC = df_selection["estadia_vs_cita_sum"].sum()
countEC = (df_selection['estadia_vs_cita_sum'] > 0).sum()

if countEC == 0:
    estadiavsCita = 0
else:
    estadiavsCita = round(sum_EC / countEC, 1)

left_column1, left_column2, right_column1, right_column2 = st.columns([1, 2, 1, 2])
with left_column1:
    st_lottie(lottie_truck, height=200, key="time")
with left_column2:
    st.subheader("Estadia vs Arribo:")
    st.subheader(f" {estadiavsArribo} hrs.")
    st.subheader(f" Placas: {countEA}")
with right_column1:
    st_lottie(lottie_list, height=200, key="list")
with right_column2:
    st.subheader("Estadia vs Cita")
    st.subheader(f"{estadiavsCita} hrs.")
    st.subheader(f" Placas: {countEC}")

st.markdown("""---""")

# ------------------------------Indicadores zonas--------------

left_column, mid_column, right_column = st.columns(3)
with left_column:
    st.write(" ")
with mid_column:
    st.markdown("<h1 style='text-align: center; color: white;'>Top 5 General</h1>", unsafe_allow_html=True)
with right_column:
    st.write(" ")

with st.container():
    left_column, mid_column1, mid_column2, mid_column3, right_column = st.columns(5)
    with left_column:
        hc.info_card(title=topdfnew['DESTINO'][0], content=f"{topdfnew['mean'][0]} hrs.",
                     theme_override=theme_bad, bar_value=100, key="top1")
        unsafe_allow_html = True
    with mid_column1:
        hc.info_card(title=topdfnew['DESTINO'][1], content=f"{topdfnew['mean'][1]} hrs.",
                     theme_override=theme_bad, bar_value=95, key="top2")
    with mid_column2:
        hc.info_card(title=topdfnew['DESTINO'][2], content=f"{topdfnew['mean'][2]} hrs.",
                     theme_override=theme_bad, bar_value=90, key="top3")
    with mid_column3:
        hc.info_card(title=topdfnew['DESTINO'][3], content=f"{topdfnew['mean'][3]} hrs.",
                     theme_override=theme_bad, bar_value=85, key="top4")
    with right_column:
        hc.info_card(title=topdfnew['DESTINO'][4], content=f"{topdfnew['mean'][4]} hrs.",
                     theme_override=theme_bad, bar_value=80, key="top5")

with st.container():
    left_column, mid_column1, mid_column2, mid_column3, right_column = st.columns(5)
    with left_column:
        st.write(f"Pendiente Descarga: {df_top1_pend_out}")
        st.write(f"Descargando: {df_top1_desc_out}")
        st.write(f"T. Ida: {df_top1_tida_out}")
        with st.expander("Placas"):
            st.dataframe(placasT1)
    with mid_column1:
        st.write(f"Pendiente Descarga: {df_top2_pend_out}")
        st.write(f"Descargando: {df_top2_desc_out}")
        st.write(f"T. Ida: {df_top2_tida_out}")
        with st.expander("Placas"):
            st.dataframe(placasT2)
    with mid_column2:
        st.write(f"Pendiente Descarga: {df_top3_pend_out}")
        st.write(f"Descargando: {df_top3_desc_out}")
        st.write(f"T. Ida: {df_top3_tida_out}")
        with st.expander("Placas"):
            st.dataframe(placasT3)
    with mid_column3:
        st.write(f"Pendiente Descarga: {df_top4_pend_out}")
        st.write(f"Descargando: {df_top4_desc_out}")
        st.write(f"T. Ida: {df_top4_tida_out}")
        with st.expander("Placas"):
            st.dataframe(placasT4)
    with right_column:
        st.write(f"Pendiente Descarga: {df_top5_pend_out}")
        st.write(f"Descargando: {df_top5_desc_out}")
        st.write(f"T. Ida: {df_top5_tida_out}")
        with st.expander("Placas"):
            st.dataframe(placasT5)

st.markdown("""---""")

# --------------------------Estadia Arribo[BAR CHART]-------------------
dfesArr = df_selection.loc[df_selection['estadia_vs_arribo_sum'] > 0]
estadia_arribo = dfesArr.groupby(by=["DESTINO"]).mean()[["estadia_vs_arribo_sum"]].round(2)
# filtro = estadia_arribo['estadia_vs_arribo_sum'] > 0
# estadia_arribo = estadia_arribo[filtro]
estadia_arribo = estadia_arribo.sort_values('estadia_vs_arribo_sum', ascending=False)
grafico_estadia_arribo = px.bar(
    estadia_arribo,
    x=estadia_arribo.index,
    y="estadia_vs_arribo_sum",
    title="<b>Estadia vs Arribo</b>",
    color_discrete_sequence=["#1E90FF"] * len(estadia_arribo),
    template="plotly_white", text_auto=True, labels={'estadia_vs_arribo_sum': 'Horas Promedio Arribo'}, height=500
)
grafico_estadia_arribo.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False)),
)

# ----------------------------Estadia Cita[BAR CHART]-------------------
dfesCit = df_selection.loc[df_selection['estadia_vs_cita_sum'] > 0]
estadia_cita = dfesCit.groupby(by=["DESTINO"]).mean()[["estadia_vs_cita_sum"]].round(2)
filtro = estadia_cita['estadia_vs_cita_sum'] > 0
estadia_cita = estadia_cita[filtro]
estadia_cita = estadia_cita.sort_values('estadia_vs_cita_sum', ascending=False)
grafico_estadia_cita = px.bar(
    estadia_cita,
    x=estadia_cita.index,
    y="estadia_vs_cita_sum",
    title="<b>Estadia vs Cita </b>",
    color_discrete_sequence=["#1E90FF"] * len(estadia_cita),
    template="plotly_white", text_auto=True, labels={'estadia_vs_cita_sum': 'Horas Promedio Cita'}, height=500
)
grafico_estadia_cita.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False)),
)
# ---- ----------------Tiempos por zona------------------------
colors = ['Orange', 'OrangeRed', 'Khaki', 'YellowGreen']

estadia_zona = dfesArr.groupby(by=["ZONA_VENTA"]).mean()[["estadia_vs_arribo_sum"]].round(2)
grafico_zona = go.Figure(
    go.Pie(labels=estadia_zona.index, values=estadia_zona["estadia_vs_arribo_sum"], hoverinfo="label+percent",
           textinfo="label+value", hole=0.5, title="<b>Zonas</b>",
           marker=dict(colors=colors, line=dict(color='#000000', width=2))))

# ------------------------Estadia Arribo Acumulado[BAR CHART]-----------------------
estadia_arribo_acum = df_selection.groupby(by=["DESTINO"]).sum()[["estadia_vs_arribo_acum"]].round(2)
grafico_estadia_arribo_acum = px.bar(
    estadia_arribo_acum,
    x=estadia_arribo_acum.index,
    y="estadia_vs_arribo_acum",
    title="<b>Estadia vs Arribo</b>",
    color_discrete_sequence=["#1E90FF"] * len(estadia_arribo_acum),
    template="plotly_white", text_auto=True, labels={'estadia_vs_arribo_acum': 'Horas Acumuladas Arribo'}, height=500
)
grafico_estadia_arribo_acum.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False)),
)

left_column, mid_column, right_column = st.columns(3)
left_column.plotly_chart(grafico_estadia_arribo, use_container_width=True)
mid_column.plotly_chart(grafico_estadia_cita, use_container_width=True)
right_column.plotly_chart(grafico_zona, use_container_width=True)
st.markdown("""---""")

left_column, mid_column, right_column = st.columns(3)
with left_column:
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name=f'df_{today}.csv',
        mime='text/csv',
    )
with mid_column:
    st.text(ultModi)

# grouped_multiple = dfesArr.groupby(['DESTINO', 'TC REAL']).agg({'estadia_vs_arribo_acum': ['mean', 'min', 'max']})
# grouped_multiple.columns = ['age_mean', 'age_min', 'age_max']
# grouped_multiple = grouped_multiple.reset_index()
# #placas = dfesArr.groupby(["DESTINO"])
# st.dataframe(grouped_multiple)


# ------------------------- HIDE STREAMLIT STYLE ------------------------
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
