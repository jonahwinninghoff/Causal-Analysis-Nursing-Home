# Libraries
library(dplyr)
library(ggplot2)

# Read the ecdf csv
directory= paste(getwd(),'Report/ecdf',sep='/')
df <- read.csv(directory)
df <- select(df, empirical, theoretical)%>%
  mutate(group='empirical')%>%
  select(empirical, group)
names(df) <- c('score','group')

# Generate theoretical beta distribution
mx <- mean(df$score)
v <- var(df$score)
alpha = abs(mx*((mx*(1-mx))/(v-1)))
beta = abs((1-mx)*((mx*(1-mx))/(v-1)))
theoretical <- data.frame(rbeta(100000,alpha,beta))
theoretical <- mutate(theoretical,group='theoretical')
names(theoretical)<-c('score','group')

# Row binding
compare <- rbind(df,theoretical)

# Make plot
ggplot(data = compare, aes(x = score, fill=group, color=group)) +
  geom_density(alpha = 0.5,bw=0.009) +
  labs(fill = "", color = "")+
  theme(panel.background = element_rect(fill = 'white'),
        panel.grid.minor = element_line(color = '#c7f3ff'),
        panel.grid.major = element_line(color = '#edfbff'),
        axis.title.y = element_text(vjust = .5, color='dimgrey',size=12),
        axis.title.x = element_text(hjust = .5,color='dimgrey',size=12),
        legend.position="top",
        legend.text = element_text(color='dimgrey',size=12))+
  scale_fill_manual(values = c('#fcad03','#00a8f0'))+
  scale_color_manual(values = c('#fcad03','#00a8f0'))

# Save this plot
ggsave("densityplot.pdf",width = 6,   height = 4)
ggplot(data = compare, aes(x = score, fill=group, color=group)) +
  geom_density(alpha = 0.5,bw=0.009) +
  labs(fill = "", color = "")+
  theme(panel.background = element_rect(fill = 'white'),
        panel.grid.minor = element_line(color = '#c7f3ff'),
        panel.grid.major = element_line(color = '#edfbff'),
        axis.title.y = element_text(vjust = .5, color='dimgrey',size=12),
        axis.title.x = element_text(hjust = .5,color='dimgrey',size=12),
        legend.position="top",
        legend.text = element_text(color='dimgrey',size=12))+
  scale_fill_manual(values = c('#fcad03','#00a8f0'))+
  scale_color_manual(values = c('#fcad03','#00a8f0'))
dev.off()

# Read non_parametric tests
directory= paste(getwd(),'nonparametric_tests',sep='/')
df <- read.csv(directory)

# Manipulate data for plot
df <- select(df, Variable, Difference.Value, Standard.Error, p_value)%>%
  mutate(CI_lower = Difference.Value - qnorm(0.05/(2*13),lower.tail = F)*Standard.Error)%>%
  mutate(CI_high = Difference.Value + qnorm(0.05/(2*13),lower.tail = F)*Standard.Error)

# Make plot
ggplot(df, aes(x = reorder(Variable,Difference.Value,FUN=mean), y=Difference.Value)) + 
  geom_point(position=position_dodge(width=0.75),color='#fcad03',alpha=0.7,size=3.7)+
  geom_abline(intercept = 0, slope = 0, lty=2,color='#00a8f0')+
  geom_errorbar(aes(ymin=CI_lower, ymax=CI_high), width=0.4,
                position=position_dodge(width=0.75),color='#00a8f0',size=0.5)+
  coord_flip()+
  labs(y="Expected difference (quality score) with 99% CI", x=NULL)+
  theme(panel.background = element_rect(fill = 'white'),
       panel.grid.minor = element_line(color = '#c7f3ff'),
       panel.grid.major = element_line(color = '#edfbff'),
       axis.title.x = element_text(vjust = .5, color='dimgrey',size=12),
       axis.text.x = element_text(colour = 'dimgrey',size=12),
       axis.text.y = element_text(colour = 'dimgrey',size=12))

## Save this plot
ggsave("differplot.pdf",width = 6,   height = 4)

ggplot(df, aes(x = reorder(Variable,Difference.Value,FUN=mean), y=Difference.Value)) + 
  geom_point(position=position_dodge(width=0.75),color='#fcad03',alpha=0.7,size=3.7)+
  geom_abline(intercept = 0, slope = 0, lty=2,color='#00a8f0')+
  geom_errorbar(aes(ymin=CI_lower, ymax=CI_high), width=0.4,
                position=position_dodge(width=0.75),color='#00a8f0',size=0.5)+
  coord_flip()+
  labs(y="Expected difference (quality score) with 99% CI", x=NULL)+
  theme(panel.background = element_rect(fill = 'white'),
        panel.grid.minor = element_line(color = '#c7f3ff'),
        panel.grid.major = element_line(color = '#edfbff'),
        axis.title.x = element_text(vjust = .5, color='dimgrey',size=12),
        axis.text.x = element_text(colour = 'dimgrey',size=12),
        axis.text.y = element_text(colour = 'dimgrey',size=12))
dev.off()

# Read actionable report
directory <- paste(getwd(),'Report/actionable',sep='/')
df <- read.csv(directory)

# Manipulate data for plot
df$X <- c('cons','zip','bed','repor','compl','fine','fam','amou','hr','lpn','nur_aid','rn')

# Make plot
ggplot(df, aes(x = reorder(X,coef,FUN=mean), y=coef)) + 
  geom_point(position=position_dodge(width=0.75),color='#fcad03',alpha=0.7,size=3.7)+  
  geom_errorbar(aes(ymin=X0.025, ymax=X0.975), width=0.4,
                position=position_dodge(width=0.75),color='#00a8f0',size=0.5) +
  geom_abline(intercept = 0, slope = 0, lty=2,color='#00a8f0')+
  coord_flip()+
  labs(y="Coefficients with 95% confidence intervals", x=NULL)+
  theme(panel.background = element_rect(fill = 'white'),
        panel.grid.minor = element_line(color = '#c7f3ff'),
        panel.grid.major = element_line(color = '#edfbff'),
        axis.title.x = element_text(vjust = .5, color='dimgrey',size=12),
        axis.text.x = element_text(colour = 'dimgrey',size=12),
        axis.text.y = element_text(colour = 'dimgrey',size=12))

## Save this plot
ggsave("causal.pdf",width = 6,   height = 4)

ggplot(df, aes(x = reorder(X,coef,FUN=mean), y=coef)) + 
  geom_point(position=position_dodge(width=0.75),color='#fcad03',alpha=0.7,size=3.7)+  
  geom_errorbar(aes(ymin=X0.025, ymax=X0.975), width=0.4,
                position=position_dodge(width=0.75),color='#00a8f0',size=0.5) +
  geom_abline(intercept = 0, slope = 0, lty=2,color='#00a8f0')+
  coord_flip()+
  labs(y="Coefficients with 95% confidence intervals", x=NULL)+
  theme(panel.background = element_rect(fill = 'white'),
        panel.grid.minor = element_line(color = '#c7f3ff'),
        panel.grid.major = element_line(color = '#edfbff'),
        axis.title.x = element_text(vjust = .5, color='dimgrey',size=12),
        axis.text.x = element_text(colour = 'dimgrey',size=12),
        axis.text.y = element_text(colour = 'dimgrey',size=12))
dev.off()

# Check endoogeneity
library(ivprobit)
directory<-paste(getwd(),'foreconometrics',sep='/')
df <- read.csv(directory)
df <- subset(df, select = -c(X,const))
names(df) <- c('ins_rat', 'zip', 'bed','repor',
               'compl','fine','abu','fam','sprin',
               'amo_fine','hr','lpn','nur_aid',
               'rn')
result <-ivprobit(ins_rat ~ zip + bed + repor + compl + fam + amo_fine + lpn + nur_aid + rn + hr | abu,df)




