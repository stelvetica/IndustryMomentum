##############################################part1
rm(list =ls())
library(readxl)
library(psymonitor)

df_initial <- read_excel('E:/专题报告/泡沫检测/泡沫代码和数据/数据更新_中信一级_快照.xlsx',sheet = "1-收盘价")#########修改位置 4-周成交额 1-收盘价
#cols<- colnames(df)[2:30]
cols<- colnames(df_initial)[2:ncol(df_initial)]
df = df_initial[2:nrow(df_initial),2:ncol(df_initial)]
dateline = df_initial[2:nrow(df_initial),1]

IC       <- 2  # use BIC to select the number of lags
adflag   <- 6  # set the maximum nuber of lags to 6
#yr       <- 2  
#Tb       <- 12*yr + swindow0 - 1  # Set the control sample size
#nboot    <- 99
#psymatrix<-matrix(0,nrow=obs-swindow0+1,ncol=42)

psymatrix<-df*NA

source('E:/专题报告/泡沫检测/泡沫代码和数据/PSY20220430/psymonitor_0.0.2/psymonitor/R/PSY.R')
#source("C:\\chart.Correlation.r")

#ncol(df)
for (col in (1:ncol(df))){
  print(col)
  y<-df[[cols[col]]]
  obs     <- length(y)
  swindow0 <- floor(obs * (0.01 + 1.8 / sqrt(obs))) # set minimal window size  62
  
  psymatrix[nrow(psymatrix)-obs+swindow0:nrow(psymatrix),col] <-PSY(y, swindow0 = swindow0, IC = IC,
                                           adflag = adflag)  # estimate the PSY test statistics sequence
}

output <-cbind(dateline,psymatrix)

write.csv(output, file = "E:/专题报告/泡沫检测/泡沫代码和数据/PSY20220430/PSY一级成交价20230922.csv")#########修改储存位置

