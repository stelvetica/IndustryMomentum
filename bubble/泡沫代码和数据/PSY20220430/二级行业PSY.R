##############################################part1
rm(list =ls())
library(readxl)
library(psymonitor)

df_initial <- read_excel('E:/专题报告/泡沫检测/泡沫代码和数据/数据更新_中信二级_快照_2.xlsx',sheet = "4-周成交额")#########修改位置
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

psymatrix <- df*NA


#source('E:/专题报告/泡沫检测/泡沫代码和数据/PSY20220430/psymonitor_0.0.2/psymonitor/R/PSY.R')

for (col in (1:ncol(df))){
  print(col)
  y<-df[[cols[col]]]
  nan_num_vec = sum(is.na(y))+1
  obs     <- length(y[nan_num_vec:nrow(df)])
  swindow0 <- floor(obs * (0.01 + 1.8 / sqrt(obs))) # set minimal window size   62
  a = nrow(psymatrix)-obs+swindow0
  b = nrow(psymatrix)
  
  psymatrix[a:b,col] <- PSY(y[nan_num_vec:nrow(df)], swindow0 = swindow0, IC = IC,
                                                                     adflag = adflag)  # estimate the PSY test statistics sequence

  
  
  }

output <-cbind(dateline,psymatrix)

write.csv(output, file = "E:/专题报告/泡沫检测/泡沫代码和数据/PSY20220430/PSY二级成交额20230331_2.csv")#########修改储存位置



##############################################part1
rm(list =ls())
library(readxl)
library(psymonitor)

df_initial <- read_excel('E:/专题报告/泡沫检测/泡沫代码和数据/数据更新_中信二级_快照.xlsx',sheet = "5-周度历史波动率")#########修改位置
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

psymatrix <- df*NA

for (col in (1:ncol(df))){
  print(col)
  y<-df[[cols[col]]]
  nan_num_vec = sum(is.na(y))+1
  obs     <- length(y[nan_num_vec:nrow(df)])
  swindow0 <- floor(obs * (0.01 + 1.8 / sqrt(obs))) # set minimal window size
  a = nrow(psymatrix)-obs+swindow0
  b = nrow(psymatrix)
  
  psymatrix[a:b,col] <- PSY(y[nan_num_vec:nrow(df)], swindow0 = swindow0, IC = IC,
                            adflag = adflag)  # estimate the PSY test statistics sequence
}

output <-cbind(dateline,psymatrix)

write.csv(output, file = "E:/专题报告/泡沫检测/PSY20220430/PSY二级波动率.csv")#########修改储存位置