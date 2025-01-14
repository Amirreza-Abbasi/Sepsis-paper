import pandas as pd
import GEOparse


import sys
sys.path.append('D:/Apps/')
import Amirreza_Python_Library as al


Zlog_df = pd.read_excel('selected genes\zlog.xlsx', header=0)

# Filter out rows with float values in 'Gene.symbol' column (drop float values to keep only Gene with valid symbols)
Zlog_List = Zlog_df[~Zlog_df['Gene.symbol'].astype(str).str.isnumeric()].reset_index(drop=True)
Zlog_List = al.Separate_Genes(Zlog_List, 'Gene.symbol')
Zlog_List = Zlog_List[['Gene.symbol']]


TFgenes_df = pd.read_excel('TF-miRNA-Gene\TF-miRNA-Gene.xlsx', header=0, index_col=0)
# Filter out rows with float values in 'Gene.symbol' column (drop float values to keep only Gene with valid symbols)
TFgenes_List = TFgenes_df[~TFgenes_df['Target'].astype(str).str.isnumeric()].reset_index(drop=True)
TFgenes_List = al.Separate_Genes(TFgenes_List, 'Target')
TFgenes_List = TFgenes_List[['Target']]


# Dataset_Code = 'GSE28750' # GPL = 'GPL570' # Title_Columnn = 'title', Symbol_Column = 'Gene Symbol'
# Dataset_Code = 'GSE13904' # GPL = 'GPL570' # Title_Columnn = 'title', Symbol_Column = 'Gene Symbol'
# Dataset_Code = 'GSE54514' # GPL = 'GPL6947' # Title_Columnn = 'source_name_ch1', Symbol_Column = 'Symbol'
# Dataset_Code = 'GSE26440' # GPL = 'GPL570' # Title_Columnn = 'source_name_ch1', Symbol_Column = 'Gene Symbol''
# Dataset_Code = 'GSE26378' # GPL = 'GPL4133' # Title_Columnn = 'source_name_ch1'

Dataset_Code = 'GSE26378'
GPL = 'GPL570'
Title_Columnn = 'title'
Symbol_Column = 'Gene Symbol'
# Download the GSE dataset
gse = GEOparse.get_GEO(Dataset_Code, destdir ='./Datasets/', annotate_gpl=True)
gse_samples = gse.phenotype_data[[Title_Columnn, "source_name_ch1"]]

temp = GEOparse.GEOTypes.BaseGEO.get_type(gse)

dir(gse)
gse.gsms
gsms_table = gse.gsms[gse_samples.index[0]].table   # a sample name to get the information for samples
gsms_Columns = gse.gsms[gse_samples.index[0]].columns

temp = gse.gpls
gpl_table = gse.gpls[GPL].table
gpl_Columns = gse.gpls[GPL].columns

Sample_Groups = gse_samples[[Title_Columnn]]
# Sample_Groups = Sample_Groups.rename(columns={'title': 'Group'})
Sample_Groups = Sample_Groups.rename(columns={Sample_Groups.columns[0]: 'Group'})
Sample_Groups = Sample_Groups[Sample_Groups['Group'].notna() & (Sample_Groups['Group'] != '')]


pivoted_control_samples = gse.pivot_samples('VALUE').reset_index()
pivoted_control_samples = gse.pivot_and_annotate('VALUE',gse.gpls[list(gse.gpls)[0]],Symbol_Column).reset_index()

# Drop rows where 'Gene Symbol' is NaN
pivoted_control_samples = pivoted_control_samples.dropna(subset=[Symbol_Column])

# only kepp genes that have more than 2 character length
pivoted_control_samples = pivoted_control_samples[pivoted_control_samples[Symbol_Column].str.len() > 2]

# convert data type to string
pivoted_control_samples[Symbol_Column] = pivoted_control_samples[Symbol_Column].astype(str)

# df = pivoted_control_samples.iloc[:10, :]
df = al.Separate_Multiple_Genes(pivoted_control_samples, Symbol_Column)


# temp = pivoted_control_samples[Symbol_Column]
df.rename(columns={Symbol_Column: 'Gene Symbol'}, inplace=True)


# =============================================================================
# Make datasets
# =============================================================================
# Filter df based on values in Zlog_List's 'Gene.symbol' column
Z_df = df[df['Gene Symbol'].isin(Zlog_List['Gene.symbol'])]
Z_df = Z_df.drop_duplicates(subset=['Gene Symbol'])

Z_df = Z_df.drop(columns=['ID_REF'])
transposed_df = Z_df.set_index('Gene Symbol').T.astype('float32')
Z_df = pd.concat([Sample_Groups, transposed_df], axis=1, join='inner')
Z_df.to_csv(f'./Datasets/{Dataset_Code}_Zlog_Ready_to_Train.csv', index=1)
# Z_df = pd.read_csv('./Datasets/{Dataset_Code}_Zlog_Ready_to_Train.csv', header=0, index_col=0)

# Filter df based on values in TFgenes_List's 'Gene.symbol' column
Target_df = df[df['Gene Symbol'].isin(TFgenes_List['Target'])]
Target_df = Target_df.drop_duplicates(subset=['Gene Symbol'])

Target_df = Target_df.drop(columns=['ID_REF'])
transposed_df = Target_df.set_index('Gene Symbol').T.astype('float32')
Target_df = pd.concat([Sample_Groups, transposed_df], axis=1, join='inner')
Target_df.to_csv(f'./Datasets/{Dataset_Code}_Target_Genes_Ready_to_Train.csv', index=1)
# Z_df = pd.read_csv('./Datasets/{Dataset_Code}_Target_Genes_Ready_to_Train.csv', header=0, index_col=0)

















