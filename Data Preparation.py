import os
import glob
from os import listdir
from os.path import isdir, join
import pandas as pd
from ipywidgets import IntProgress
from IPython.display import display

def progress_bar(dirs,path,extension,data):
    progress = IntProgress()
    progress.max = len(mydirs)
    progress.description = '(Init)'
    display(progress)
    for mydir in dirs:
        os.chdir(path + mydir)
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
        for file in all_filenames:
            df_temp = pd.read_csv(file, encoding='utf8')
            try:
                data = pd.merge(data, df_temp, how='outer', on='SEQN')
            except:
                print(mypath + mydir + '/' + file)
        progress.value += 1
        progress.description = mydir
    progress.description = '(Done)'


def merge_csv(mypath, dff):
    mydirs = [f for f in listdir(mypath) if isdir(join(mypath, f))]
    mydirs.remove("Demographics")
    mydirs.remove("Dietary")
    datapath = mypath + "Demographics/DEMO.csv"
    dff = pd.read_csv(datapath , encoding='utf8')
    extension = 'csv'
    progress_bar(mydirs,mypath,extension,dff)


# 2015-2016
mypath = "/Users/Tim/Desktop/scor_test/Data/2015-2016/"
mydirs = [f for f in listdir(mypath) if isdir(join(mypath, f))]
mydirs.remove("Demographics")
mydirs.remove("Dietary")
df9 = pd.read_csv("/Users/Tim/Desktop/scor_test/Data/2015-2016/Demographics/DEMO_I.csv", encoding='utf8')
extension = 'csv'

# Initialize a progess bar
progress = IntProgress()
progress.max = len(mydirs)
progress.description = '(Init)'
display(progress)
for mydir in mydirs:
    os.chdir(mypath + mydir)
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    for file in all_filenames:
        df_temp = pd.read_csv(file, encoding='utf8')
        try:
            df9 = pd.merge(df9, df_temp, how='outer',on='SEQN')
        except:
            print(mypath + mydir + '/' + file)
    progress.value += 1
    progress.description = mydir
progress.description = '(Done)'

k = 0
df_list = ["df","df1","df2","df3","df4","df5","df6","df7","df8","df9"]
for dff in df_list:
    dff = pd.DataFrame()

def merge_all():
    root_path =  "/Users/Meng/Desktop/scor_test/Data/"
    my_years = ['1999-2000','2001-2002','2003-2004','2005-2006','2007-2008','2009-2010','2011-2012','2013-2014','2015-2016']
    k = 0
    for myyear in my_years:
        global k
        mypath = root_path + myyear + "/"
        merge_csv(mypath,df_list[k])
        k += 1
    listA = df.columns
    listB = df2.columns
    listC = df3.columns
    listD = df4.columns
    listE = df5.columns
    listF = df6.columns
    listG = df7.columns
    listH = df8.columns
    listI = df9.columns

    ret = list(set(listA).intersection(set(listB)))
    ret = list(set(ret).intersection(set(listC)))
    ret = list(set(ret).intersection(set(listD)))
    ret = list(set(ret).intersection(set(listE)))
    ret = list(set(ret).intersection(set(listF)))
    ret = list(set(ret).intersection(set(listG)))
    ret = list(set(ret).intersection(set(listH)))
    ret = list(set(ret).intersection(set(listI)))

    ret.remove('WTSAF2YR_y')
    ret.remove('WTSAF2YR_x')

    len(ret)

    # test it with features within all years
    my_features = ret
    selected_df = df[my_features]
    selected_df1 = df1[my_features]
    selected_df2 = df2[my_features]
    selected_df3 = df3[my_features]
    selected_df4 = df4[my_features]
    selected_df5 = df5[my_features]
    selected_df6 = df6[my_features]
    selected_df7 = df7[my_features]
    selected_df8 = df8[my_features]
    finaldf = pd.concat([selected_df,selected_df1,selected_df2,selected_df3,selected_df4,selected_df5,selected_df6,selected_df7,selected_df8])
    finaldf.to_csv("/Users/Meng/Desktop/scor_test/TESTCODE/all_years.csv")

def main():
    merge_all()

if __name__ == '__main__':
    main()