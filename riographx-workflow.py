import psycopg2
import time
from pyspark import *
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType,StringType,ArrayType,IntegerType
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import matplotlib.pyplot as plt
import numpy as np
import grinpy as gp
import re
from pylatexenc.latex2text import LatexNodes2Text
import sys
from fpdf import FPDF
plt.switch_backend('agg')

def pdf(dataframe, id, results):
	pdf = FPDF()
	pdf.add_font('DejaVu', '', '/opt/spark-data/DejaVuSansCondensed.ttf', uni=True)
	pdf.set_font('DejaVu', '', 10)	
	pdf.add_page()
	pdf.image('/opt/spark-data/logo.png', x=10, y=8, w=70)
	pdf.ln(40)
	n_anterior = 0
	column = np.array(dataframe.schema.names)
	data = np.array(dataframe.select("*").collect())
	#print column[0]
	
	row_height = pdf.font_size
	spacing=1.5
	pagina = 1
	for row in data:
		if (row[1].astype("int")>n_anterior):
			#pdf.add_page()
			#pdf.image('/opt/spark-data/logo.png', x=10, y=8, w=70)
			#pdf.ln(40)
			pdf.set_font('DejaVu', '', 16)
			pdf.cell(0, 10, txt="Order #{}".format(row[1]), ln=1)		
			pdf.set_font('DejaVu', '', 10)
		n_anterior = row[1].astype("int")		
		col_width = 50
		top = pdf.y
		offset = pdf.x + 50	
		for item in column:
			pdf.multi_cell(col_width, row_height*spacing,txt=item, border=1)
			#pdf.ln(row_height*spacing)
		#pdf.ln(row_height*spacing)
		pdf.y = top
		desenha=0
        
		for item in row:
			col_width = 20
			desenha+=1
			if (desenha == 1):
				draw_graph(item) 
				pdf.image("/opt/spark-data/images/"+item+".png", x=pdf.x+100, y=top, w=50)			
			elif (desenha > 4):
				item = item.astype("float")
				item = np.around(item,4)
				item = str(item)
			pdf.x = offset
			pdf.multi_cell(col_width, row_height*spacing,txt=item, border=1)
			#pdf.ln(row_height*spacing)
				
		pdf.ln(row_height*spacing)
		pagina+=1
		if (pagina == 5):
			pdf.add_page()
			pdf.image('/opt/spark-data/logo.png', x=10, y=8, w=70)
			pdf.ln(40)
			pagina = 1
			
	pdf.output('/opt/spark-data/reports/'+id+'.pdf')

def mysplit(mystr,operators):
	mystr.replace('', '')
	return re.split(operators, mystr.replace(" ", ""))

def eigenvetor(funcao,g6,n,complement):
	n = int(n)    
	if (complement=='False'):
		G = graph(g6)
	else:
		G = graph_complement(g6)
	if (funcao == 'lambda'):
		eigen = nx.adjacency_spectrum(G)
	elif (funcao == 'mu'):
		eigen = nx.laplacian_spectrum(G)
	elif (funcao == 'q'):
		eigen = nx.signless_laplacian_spectrum(G)
	elif (funcao == 'alpha'):
		if(connected(G) == True):
			eigen = nx.sign_dist_spectrum(G)
		else:
			eigen = None
	try:
		eigen = np.sort(eigen)
		eigen = eigen[::-1]	
		return float(eigen[n-1])
	except IndexError:
		return None
	except TypeError:
		return None
	
def eigenvalue(funcao,g6,complement):
	if (complement=='False'):
		G = graph(g6)
	else:
		G = graph_complement(g6)
	if (funcao == 'chi'):
		eigen = gp.chromatic_number(G)
	elif (funcao == 'omega'):
		eigen = gp.clique_number(G)
	elif (funcao == 'larg_degree'):
		eigen = max(degree_sequence(G))
	try:
	    return int(eigen)
	except IndexError:
		return None
	except TypeError:
		return None
	
def graph(g6):
    g6 = str.encode(g6)
    G = nx.from_graph6_bytes(g6)
    return G

def graph_complement(g6):
    g6 = str.encode(g6)
    G = nx.from_graph6_bytes(g6)
    GC = nx.complement(G)
    return GC

def degree_sequence(G):
    degree_sequence = []
    for x,y in G.degree():
        degree_sequence.append(y) 
    return degree_sequence

def triangle_free(G):
	return gp.is_triangle_free(G)

def bipartite(G):
	return nx.is_bipartite(G)

def connected(G):
        return nx.is_connected(G)

def draw_graph(g6):
    g6b = str.encode(g6)
    G = nx.from_graph6_bytes(g6b)
    nx.draw(G, with_labels=True)
    plt.savefig("/opt/spark-data/images/"+g6+".png")
    plt.clf()



def workflow(df,ordem1,ordem2,results,trianglefree,conexo,bipartido,latex,id,grau1,grau2,min_max):
	start_time = time.time()
	exp = latex.replace("\alpha","α")
	exp = exp.replace("overline","bar")
	exp = LatexNodes2Text().latex_to_text(exp)
	exp = exp.replace('̅','bar')

	#First filter
	df = df.filter( (df.n >= ordem1) & (df.n <= ordem2) & (df.min_degree >= grau1) & (df.max_degree <= grau2)).cache()

	#Second filter
	if (trianglefree == 'True'):
		df = df.where(df.triangle_free == 1)

	if (conexo == 'True'):
		df = df.where(df.connected == 1)

	if (bipartido == 'True'):
		df = df.where(df.bipartite == 1)

	#latex treat
	exp1 = exp.replace(')','')
	exp1 = exp1.replace('(','')
	split = mysplit(exp1,"[+-/*]")

	#Calculating eigenvalues
	for sp in split:		
		if (sp.find('λbar')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvector(lit('lambda'),'g6_code',lit(sp2[1]),lit('True')),4)).cache()
		elif (sp.find('μbar')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvector(lit('mu'),'g6_code',lit(sp2[1]),lit('True')),4)).cache()
		elif (sp.find('qbar')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvector(lit('q'),'g6_code',lit(sp2[1]),lit('True')),4)).cache()
		elif (sp.find('χbar')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvalue(lit('chi'),'g6_code',lit('True')),4)).cache()
		elif (sp.find('ωbar')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvalue(lit('omega'),'g6_code',lit('True')),4)).cache()
		elif (sp.find('Δbar')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvalue(lit('larg_degree'),'g6_code',lit('True')),4)).cache()
		elif (sp.find('αbar')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvector(lit('alpha'),'g6_code',lit(sp2[1]),lit('True')),4)).cache()		
		elif (sp.find('λ')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvector(lit('lambda'),'g6_code',lit(sp2[1]),lit('False')),4)).cache()
		elif (sp.find('μ')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvector(lit('mu'),'g6_code',lit(sp2[1]),lit('False')),4)).cache()
		elif (sp.find('q')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvector(lit('q'),'g6_code',lit(sp2[1]),lit('False')),4)).cache()
		elif (sp.find('χ')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvalue(lit('chi'),'g6_code',lit('False')),4)).cache() 
		elif (sp.find('ω')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvalue(lit('omega'),'g6_code',lit('False')),4)).cache() 
		elif (sp.find('Δ')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvalue(lit('larg_degree'),'g6_code',lit('False')),4)).cache()
		elif (sp.find('α')!=-1):
			sp2 = mysplit(sp,"_")
			df = df.withColumn(sp, bround(udf_eigenvector(lit('alpha'),'g6_code',lit(sp2[1]),lit('False')),4)).cache()

	#Variable change
	exp_final = exp.replace('λ','df.λ')
	exp_final = exp_final.replace('μ','df.μ')
	exp_final = exp_final.replace('q','df.q')
	exp_final = exp_final.replace('n','df.n')
	exp_final = exp_final.replace('χ','df.χ')
	exp_final = exp_final.replace('ω','df.ω')
	exp_final = exp_final.replace('Δ','df.Δ')
	exp_final = exp_final.replace('α','df.α')


	try:
		df = df.withColumn(exp,bround(eval(exp_final),4)).cache()
	except:
		df = df.withColumn(exp,None).cache()

	df = df.na.drop(subset=[exp])
	df = df.drop("triangle_free","connected","bipartite")
	
	if (min_max == 'True'):
		df = df.sort(desc(exp)).cache()
	else:
		df = df.sort(asc(exp)).cache()
	
	df5 = df.where(df.n == ordem1).limit(results)
	order = ordem1 + 1
	while(order <= ordem2):
		df5 = df5.unionAll(df.where(df.n == order).limit(results))
		order+=1
	
	pdf(df5,id,results)
	print("--- %s seconds ---" % (time.time() - start_time))
	df.unpersist

#SPARK	environment
conf = (SparkConf()
        .setAppName("RioGraphX")
        .setMaster('spark://spark-master:7077')
        .set('spark.driver.memory', '')
		.set('spark.executor.memory', '')
		.set('spark.driver.cores', '')
		.set('spark.cores.max', '')
		
        )
context = SparkContext(conf=conf)
context.setLogLevel("ERROR")
spark = SparkSession(context)

#Load dataset
dataframe = spark.read.load( "/opt/spark-data/graphs.csv", format="csv", sep="," ) \
	  .toDF("g6_code","n","min_degree","max_degree","triangle_free","connected","bipartite" )  

#Declare UDF´s
spark.udf.register("", eigenvetor, FloatType())
spark.udf.register("", eigenvalue, IntegerType())
udf_eigenvector = udf(eigenvetor, FloatType())
udf_eigenvalue = udf(eigenvalue, IntegerType())

#Database connection
connection = psycopg2.connect(user="", password="", host="", port="5432", database="graphx")
cursor = connection.cursor()

postgreSQL_select_Query = "select * from submit where spark_ok = 'false' LIMIT 1;"
postgreSQL_update_Query = "update submit set spark_ok = 'true' where id_submit = %s ;"
postgreSQL_update_Result1 = "update results set status = 'PROCESSING', start_datetime = now() where id_submit = %s ;"
postgreSQL_update_Result2 = "update results set status = 'FINISHED', end_datetime=now() where id_submit = %s ;"
postgreSQL_update_Result3= "update results set status = 'ERROR', end_datetime=now() where id_submit = %s ;"

#Main 
while(1):
    cursor.execute(postgreSQL_select_Query)
    records = cursor.fetchall()
    if(records):
        for row in records:
            id_submit = int(row[0])
            print('Executando submissao', id_submit)
            cursor.execute(postgreSQL_update_Query, (id_submit,))
            cursor.execute(postgreSQL_update_Result1, (id_submit,))
            connection.commit()
            try:
                print('Iniciando processo SPARK')
                workflow(dataframe,int(row[3]),int(row[4]),int(row[7]),str(row[8]),str(row[9]),str(row[10]),str(row[1]),str(id_submit),int(row[5]),int(row[6]),str(row[2]))
                spark.catalog.clearCache()
                print('Terminando processo SPARK')
                cursor.execute(postgreSQL_update_Result2, (id_submit,))
                connection.commit()
            except:
                cursor.execute(postgreSQL_update_Result3, (id_submit,))
                connection.commit()
    connection.commit()