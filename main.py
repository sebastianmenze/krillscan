from random import random

from bokeh.layouts import column, row
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.tile_providers import CARTODBPOSITRON, get_provider, OSM
from bokeh.models import CDSView, ColumnDataSource, IndexFilter
# from bokeh.document import without_document_lock

# import geoviews as gv
# import geoviews.feature as gf
# from geoviews import dim, opts

# gv.extension('bokeh')
from pyproj import Transformer

import shutil
import pathlib

from bokeh.io import show
from bokeh.models import CustomJS, Toggle, FileInput
# from bokeh.io import curdoc
from threading import Timer

import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

from skimage.transform import  resize

from echolab2.instruments import EK80, EK60
import configparser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os
import logging

from scipy.ndimage.filters import uniform_filter1d

from scipy.signal import convolve2d
from scipy.interpolate import interp1d
# from skimage.transform import  resize

from echopy import transform as tf
from echopy import resample as rs
from echopy import mask_impulse as mIN
from echopy import mask_seabed as mSB
from echopy import get_background as gBN
from echopy import mask_range as mRG
from echopy import mask_shoals as mSH
from echopy import mask_signal2noise as mSN

from skimage.transform import rescale, resize 

from pyproj import Geod
geod = Geod(ellps="WGS84")

import sys
import matplotlib
# matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets

# from PyQt5.QtWidgets import QShortcut
# from PyQt5.QtGui import QKeySequence
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import qdarktheme
import time

from pathlib import Path

from pykrige.ok import OrdinaryKriging
import cartopy.crs as ccrs
import cartopy as cart

from matplotlib.colors import ListedColormap
import re
import traceback
from pyproj import Proj, transform
import zipfile

import smtplib
import ssl
# import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.text  import MIMEText

#%%

        
class Worker(QtCore.QThread):
 
    # def __init__(self, *args, **kwargs):

        
    def scan_folder(self):

            # self.workpath=  os.path.join(self.folder_source,'krill_data')     
            os.chdir(self.workpath)
        
            new_df_files = pd.DataFrame([])           
            new_df_files['path'] = glob.glob( os.path.join( self.folder_source,'*.raw') )  
        
            dates=[]
            for fname in new_df_files['path']:
                
                datetimestring=re.search('D\d\d\d\d\d\d\d\d-T\d\d\d\d\d\d',fname).group()
                dates.append( pd.to_datetime( datetimestring,format='D%Y%m%d-T%H%M%S' ) )
            new_df_files['date'] = dates
        

            new_df_files['to_do']=True 
            
            
            self.df_files=pd.concat([self.df_files,new_df_files])
            self.df_files.drop_duplicates(inplace=True)
            
            self.df_files =  self.df_files.sort_values('date')
            self.df_files=self.df_files.reset_index(drop=True)
            
            print('found '+str(len(self.df_files)) + ' raw files')
         
            
            # look for already processed data
            self.df_files['to_do']=True    
            
            if os.path.isfile('list_of_rawfiles.csv'):
                df_files_done =  pd.read_csv('list_of_rawfiles.csv',index_col=0)
                df_files_done=df_files_done.loc[ df_files_done['to_do']==False,: ]
            
                names = self.df_files['path'].apply(lambda x: Path(x).stem)       
                names_done = df_files_done['path'].apply(lambda x: Path(x).stem)       
                
            # print(names)
            # print(nasc_done)
                ix_done= names.isin( names_done  )  

            # print(ix_done)
                self.df_files.loc[ix_done,'to_do'] = False        
            self.n_todo=np.sum(self.df_files['to_do'])
            print('To do: ' + str(self.n_todo))
            
            
    def pass_source_folder(self,folder_source):
        self.folder_source=folder_source
    def pass_work_folder(self,workpath):
        self.workpath=workpath
        
    def scan_and_process(self):
        if self.not_processing:
            self.not_processing=False
            self.scan_folder()         
            self.process()
            self.not_processing=True

    def process(self):
        
        

        echogram=pd.DataFrame([])    
        positions=pd.DataFrame([])    
        
        unit_length_min=pd.to_timedelta(10,'min')

        for index, row in self.df_files.iterrows():
            if self.keepworking & (row['to_do']==True):
                rawfile=row['path']
                print('working on '+rawfile)
                try:
                    
                    # breakpoint()
                    
                    echogram_file, positions_file = self.read_raw(rawfile)
                    
                    
                    echogram = pd.concat([ echogram,echogram_file ])
                    positions = pd.concat([ positions,positions_file ])
                    t=echogram.index
                    
                    # print(echogram)
                    
                    # print( [ t.max() , t.min() ])
                    
                    while (t.max() - t.min()) > unit_length_min:
                        
                        # print(  (t.min() + unit_length_min) > t)
                        ix_end = np.where( (t.min() + unit_length_min) > t )[0][-1]
                        ix_start=t.argmin()
                        # print([ix_start,ix_end])
                        
                    # accumulate 10 min snippet  
                        new_echogram = echogram.iloc[ix_start:ix_end,:]
                        new_positions = positions.iloc[ix_start:ix_end,:]
                        echogram = echogram.iloc[ix_end::,:]
                        positions = positions.iloc[ix_end::,:]
                        t=echogram.index

                        # try:
                        df_nasc_file, df_sv_swarm = self.detect_krill_swarms(new_echogram,new_positions)   
                        name = t.min().strftime('D%Y%m%d-T%H%M%S' )         
                        
                        df_sv_swarm[ new_echogram==-999 ] =-999
                        
                                                
                        df_nasc_file.to_hdf( name + '_nasctable.h5', key='df', mode='w'  )
                        
                        dffloat=df_nasc_file.copy()
                        formats = {'lat': "{:.6f}", 'lon': "{:.6f}", 'distance_m': "{:.4f}",'bottomdepth_m': "{:.1f}",'nasc': "{:.2f}"}
                        for col, f in formats.items():
                            dffloat[col] = dffloat[col].map(lambda x: f.format(x))                           
                        # dffloat.to_csv( name + '_nasctable.gzip',compression='gzip' )
                        dffloat.to_csv( name + '_nasctable.csv')
                        
                        df_sv_swarm.astype('float16').to_hdf( name + '_sv_swarm.h5', key='df', mode='w'  )
                        # self.df_files.loc[i,'to_do'] = False
                        # except Exception as e:
                        #   print(e)                      
                    self.df_files.loc[index,'to_do']=False            
                    self.df_files.to_csv('list_of_rawfiles.csv')
                   
                except Exception as e:
                    print(e)               
                    print(traceback.format_exc())
                    breakpoint()

    def start(self):
        self.keepworking=True       
        self.not_processing=True        
        self.df_files = pd.DataFrame(columns=['path','date','to_do'])
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.scan_and_process)       
        
        logging.basicConfig(filename="logfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')
                # Creating an object
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        # print('START automatic processing')
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')   

        self.timer.start(1000)
       

    def stop(self):
        # self.keepRunning = False
        self.keepworking=False       
        # self.terminate()
        self.quit()
        # print('STOP automatic processing')

    def read_raw(self,rawfile):       
        df_sv=pd.DataFrame( [] )
        positions=pd.DataFrame( []  )
        
        # breakpoint()
        
        # print('Echsounder data are: ')
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')   
   
        try:     
            raw_obj = EK80.EK80()
            raw_obj.read_raw(rawfile)
            print(raw_obj)
        except Exception as e:            
            print(e)       
            try:     
                raw_obj = EK60.EK60()
                raw_obj.read_raw(rawfile)
                print(raw_obj)
            except Exception as e:
                print(e)       
                
                                           
        
        raw_freq= list(raw_obj.frequency_map.keys())
        
        # self.ekdata=dict()
        
        # for f in raw_freq:
        f=float(self.config['GENERAL']['transducer_frequency'])
        print(raw_freq)
     
        raw_data = raw_obj.raw_data[raw_obj.frequency_map[f][0]][0]                   
        cal_obj = raw_data.get_calibration()
        
        try: 
           cal_obj.gain=float(self.config['CALIBRATION']['gain']       )
        except:
            pass
        try: 
           cal_obj.sa_correction=float(self.config['CALIBRATION']['sa_correction']       )
        except:
            pass
        try: 
           cal_obj.beam_width_alongship=float(self.config['CALIBRATION']['beam_width_alongship']       )
        except:
            pass
        try: 
           cal_obj.beam_width_athwartship=float(self.config['CALIBRATION']['beam_width_athwartship']       )
        except:
            pass
        try: 
           cal_obj.angle_offset_alongship=float(self.config['CALIBRATION']['angle_offset_alongship']       )
        except:
            pass
        try: 
           cal_obj.angle_offset_athwartship=float(self.config['CALIBRATION']['angle_offset_athwartship']       )
        except:
            pass
            
        
        sv_obj = raw_data.get_sv(calibration = cal_obj)    
          
        positions =pd.DataFrame(  raw_obj.nmea_data.interpolate(sv_obj, 'GGA')[1] )
       
        svr = np.transpose( 10*np.log10( sv_obj.data ) )
        
        # print(svr)

       
        # r=np.arange( sv_obj.range.min() , sv_obj.range.max() , 0.5 )
        r=np.arange( 0 , sv_obj.range.max() , 0.5 )

        t=sv_obj.ping_time

        sv=  resize(svr,[ len(r) , len(t) ] )

       # print(sv.shape)
       
        # estimate and correct background noise       
        p         = np.arange(len(t))                
        s         = np.arange(len(r))          
        bn, m120bn_ = gBN.derobertis(sv, s, p, 5, 20, r, np.mean(cal_obj.absorption_coefficient) ) # whats correct absoprtion?
        b=pd.DataFrame(bn)
        bn=  b.interpolate(axis=1).interpolate(axis=0).values                        
        sv_clean     = tf.log(tf.lin(sv) - tf.lin(bn))

     # -------------------------------------------------------------------------
     # mask low signal-to-noise 
        msn             = mSN.derobertis(sv_clean, bn, thr=12)
        sv_clean[msn] = np.nan

    # get mask for seabed
        mb = mSB.ariza(sv, r, r0=20, r1=1000, roff=0,
                          thr=-38, ec=1, ek=(3,3), dc=10, dk=(5,15))
        sv_clean[mb]=-999
        
        
        
                                           
        df_sv=pd.DataFrame( np.transpose(sv_clean) )
        df_sv.index=t
        df_sv.columns=r
        
        # print(df_sv)
        # print(positions)
           
        return df_sv, positions

            
    def detect_krill_swarms(self,sv,positions):
         # sv= self.echodata[rawfile][ 120000.0] 
         # sv= self.ekdata[ 120000.0]          
         # breakpoint()
              
         t120 =sv.index
         r120 =sv.columns.values

         Sv120=  np.transpose( sv.values )
         # get swarms mask
         k = np.ones((3, 3))/3**2
         Sv120cvv = tf.log(convolve2d(tf.lin( Sv120 ), k,'same',boundary='symm'))   
 
         p120           = np.arange(np.shape(Sv120cvv)[1]+1 )                 
         s120           = np.arange(np.shape(Sv120cvv)[0]+1 )           
         m120sh, m120sh_ = mSH.echoview(Sv120cvv, s120, p120, thr=-70,
                                    mincan=(3,10), maxlink=(3,15), minsho=(3,15))

        # -------------------------------------------------------------------------
        # get Sv with only swarms
         Sv120sw =  Sv120.copy()
         Sv120sw[~m120sh] = np.nan
  
         ixdepthvalid= (r120>=20) & (r120<=500)
         Sv120sw[~ixdepthvalid,:]=np.nan
  
         
         cell_thickness=np.abs(np.mean(np.diff( r120) ))               
         nasc_swarm=4*np.pi*1852**2 * np.nansum( np.power(10, Sv120sw /10)*cell_thickness ,axis=0)   
         
         # nasc_swarm[nasc_swarm>20000]=np.nan
                         
          
         df_sv_swarm=pd.DataFrame( np.transpose(Sv120sw) )
         df_sv_swarm.index=t120
         df_sv_swarm.columns=r120
          # print('df_sv')
         
         df_nasc_file=pd.DataFrame([])
         # df_nasc_file['time']=positions['ping_time']
         df_nasc_file['lat']=positions['latitude']
         df_nasc_file['lon']=positions['longitude']
         df_nasc_file['distance_m']=np.append(np.array([0]),geod.line_lengths(lons=positions['longitude'],lats=positions['latitude']) )
         
         bottomdepth=[]
         for index, row in sv.iterrows():
             if np.sum(row==-999)>0:
                 bottomdepth.append( np.min(r120[row==-999]) )
             else:
                 bottomdepth.append( r120.max() )
            
         df_nasc_file['bottomdepth_m']=bottomdepth
            
           
         df_nasc_file['nasc']=nasc_swarm
         df_nasc_file.index=positions['ping_time']
         
         # df_nasc_file=df_nasc_file.resample('5s').mean()
         print('Krill detection complete: '+str(np.sum(nasc_swarm)) ) 
        
         return df_nasc_file, df_sv_swarm
         # print(df_nasc_file)
         # df_nasc_file = df_nasc_file
         # self.df_sv_swarm = df_sv_swarm
         # self.df_sv = sv


class Worker_email(QtCore.QThread):
    
    def pass_source_folder(self,folder_source):
        self.folder_source=folder_source
    def pass_work_folder(self,workpath):
        self.workpath=workpath

    def start(self):
        self.keepworking=True       
        self.not_processing=True        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.scan_and_send_emails)       
        
        logging.basicConfig(filename="logfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='a')
                # Creating an object
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        # print('START automatic processing')

        self.timer.start(10000)
       

    def stop(self):
        self.keepworking=False       
        self.quit()
        # print('STOP automatic processing')
        
            
    def scan_and_send_emails(self):
        
        
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')   
                
        emailfrom = self.config['EMAIL']['email_from']
        emailto = self.config['EMAIL']['email_to']
        # fileToSend = r"D20220212-T180420_nasctable.h5"
        # username = "raw2nasc"
        # password = "raw2nasckrill"
        password =self.config['EMAIL']['pw']
        
        # breakpoint()
        
        # self.workpath=  os.path.join(self.folder_source,'krill_data')
        
        os.chdir(self.workpath)
        
        nasc_done =  pd.DataFrame( glob.glob( '*_nasctable.h5' ) )
        if len(nasc_done)>0:
                                
                    
                    
            if os.path.isfile('list_of_sent_files.csv'):
                df_files_sent =  pd.read_csv('list_of_sent_files.csv',index_col=0)
                ix_done= nasc_done.iloc[:,0].isin( df_files_sent.iloc[:,0]  )  
                nasc_done=nasc_done[~ix_done]
            
            else:    
                df_files_sent=pd.DataFrame([])
                
            nascfile_times=pd.to_datetime( nasc_done.iloc[:,0] ,format='D%Y%m%d-T%H%M%S_nasctable.h5' )
            nasc_done=nasc_done.iloc[np.argsort(nascfile_times),0].values
                 
            n_files=int(self.config['EMAIL']['files_per_email'])
            send_echograms=bool(self.config['EMAIL']['send_echograms'])
            echogram_resolution_in_seconds=str(self.config['EMAIL']['echogram_resolution_in_seconds'])
            
            while (len(nasc_done)>n_files) & self.keepworking  :
                
                print( str(len(nasc_done)) )
                
                files_to_send=nasc_done[0:n_files]
                # print(nasc_done)
                
                msg = MIMEMultipart()
                msg["From"] = emailfrom
                msg["To"] = emailto
                msg["Subject"] = "Krillscan data from "+ self.config['GENERAL']['vessel_name']+' ' +files_to_send[0][0:17]+'_to_'+files_to_send[-1][0:17]
              
                msgtext = str(dict(self.config['GENERAL']))
                msg.attach(MIMEText( msgtext   ,'plain'))

                loczip = msg["Subject"]+'.zip'
                zip = zipfile.ZipFile(loczip, "w", zipfile.ZIP_DEFLATED)
                zip.write('settings.ini')

                for fi in files_to_send:   
                    zip.write(fi)                                  

                # for fi in files_to_send:                                     
                #     fp = open(fi, "rb")
                #     attachment = MIMEBase('application', 'x-zip')
                #     attachment.set_payload(fp.read())
                #     fp.close()
                #     encoders.encode_base64(attachment)
                #     attachment.add_header("Content-Disposition", "attachment", filename=fi)
                #     msg.attach(attachment)

                if send_echograms:
                    
                    
                    # df_sv=pd.DataFrame([])
                    # for fi in files_to_send:      
                    #     df=pd.read_hdf(fi[0:17] + '_sv_swarm.h5',key='df')
                    #     df_sv=pd.concat([df_sv,df])
                    # df_sv=df_sv.resample(echogram_resolution_in_seconds+'s').mean()
                    # targetname=files_to_send[0][0:17] +'_to_' + files_to_send[-1][0:17] +  '_sv_swarm_mail.h5' 
                    # df_sv.to_hdf(targetname,key='df',mode='w')               
                    # fp = open(targetname, "rb")
                    # attachment = MIMEBase('application', 'x-hdf5')
                    # attachment.set_payload(fp.read())
                    # fp.close()
                    # encoders.encode_base64(attachment)
                    # attachment.add_header("Content-Disposition", "attachment", filename=targetname)
                    # msg.attach(attachment)                                 
                    # # os.remove(targetname)
                        
                    for fi in files_to_send:      

                        # fi=        files_to_send.iloc[0,0]
                        df = pd.read_hdf(fi[0:17] + '_sv_swarm.h5' ,key='df') 
                        df=df.resample(echogram_resolution_in_seconds+'s').mean()
                        targetname=fi[0:17] + '_sv_swarm_mail.h5' 
                        df.astype('float16').to_hdf(targetname,key='df',mode='w')
                        # df.astype('float16').to_csv(targetname,compression='gzip')
                        zip.write(targetname)                                  
                   
                        # fp = open(targetname, "rb")
                        # attachment = MIMEBase('application', 'x-gzip')
                        # attachment.set_payload(fp.read())
                        # fp.close()
                        # encoders.encode_base64(attachment)
                        # attachment.add_header("Content-Disposition", "attachment", filename=targetname)
                        # msg.attach(attachment)           
                        
                        os.remove(targetname)
                
                zip.close()
                fp = open(loczip, "rb")
                attachment = MIMEBase('application', 'x-zip')
                attachment.set_payload(fp.read())
                fp.close()
                encoders.encode_base64(attachment)
                attachment.add_header("Content-Disposition", "attachment", filename=loczip)
                msg.attach(attachment)    
                
                os.remove(loczip)

                        
                ctx = ssl.create_default_context()
                server = smtplib.SMTP_SSL("smtp.gmail.com", port=465, context=ctx)
                
                server.login(emailfrom, password)
                
                # print(df_files_sent)
            
                try:
                    server.sendmail(emailfrom, emailto.split(','), msg.as_string())
                    if len(df_files_sent)>0:
                        df_files_sent= pd.concat([pd.Series(df_files_sent.iloc[:,0].values),pd.DataFrame(files_to_send)],ignore_index=True)
                    else:
                        df_files_sent=pd.DataFrame(files_to_send)
                        
                    # df_files_sent=df_files_sent.reset_index(drop=True)
                    df_files_sent=df_files_sent.drop_duplicates()
                    df_files_sent.to_csv('list_of_sent_files.csv')
                    
                    
                    print('email sent: ' +   msg["Subject"] )
                    nasc_done=nasc_done[n_files::]

                except Exception as e:
                    print(e)
                                        
                server.quit()
        
#%%

# # create a plot and style its properties
# p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
# p.border_fill_color = 'black'
# p.background_fill_color = 'black'
# p.outline_line_color = None
# p.grid.grid_line_color = None

# # add a text renderer to the plot (no data yet)
# r = p.text(x=[], y=[], text=[], text_color=[], text_font_size="26px",
#            text_baseline="middle", text_align="center")

# i = 0

# ds = r.data_source

# # create a callback that adds a number in a random location
# def callback():
#     global i

#     # BEST PRACTICE --- update .data in one step with a new dict
#     new_data = dict()
#     new_data['x'] = ds.data['x'] + [random()*70 + 15]
#     new_data['y'] = ds.data['y'] + [random()*70 + 15]
#     new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i%3]]
#     new_data['text'] = ds.data['text'] + [str(i)]
#     ds.data = new_data

#     i = i + 1
    
app = QtWidgets.QApplication(sys.argv)
app.setApplicationName("Krillscan")    
    
# fname=r'D:\2022807-ANTARCTIC-PROVIDER'
    
thread = QtCore.QThread()         
w = Worker()
# w.pass_folder(fname)   
w.moveToThread(thread)

thread_email = QtCore.QThread()         
w_email = Worker_email()
# w_email.pass_folder(fname)   
w_email.moveToThread(thread_email)
    
# def stop_processing():       
#     thread.quit()
#     thread.wait() 

def settings_default(folder_source):          
    config = configparser.ConfigParser()
    config['GENERAL'] = {'source_folder': folder_source,
                         'transducer_frequency': 120000.0,
                         'vessel_name': 'MS example'}
    config['CALIBRATION'] = {'gain': 'None',
                         'sa_correction': 'None',
                         'beam_width_alongship':'None',
                         'beam_width_athwartship':'None',
                         'angle_offset_alongship':'None',
                         'angle_offset_athwartship':'None'}
    config['GRAPHICS'] = {'sv_min': -80,
                         'sv_max': -40,
                         'min_speed_in_knots': 5,
                         'nasc_map_max': 10000,
                         'nasc_graph_max': 50000}
                         
    config['EMAIL'] = {'email_from': "raw2nasc@gmail.com",
                         'email_to': "raw2nasc@gmail.com",
                         'pw': "myxdledwtfwuezis",
                         'files_per_email': 6*4,
                         'send_echograms': False,
                         'echogram_resolution_in_seconds': 60}
   
    with open('settings.ini', 'w') as configfile:
      config.write(configfile)        
          
# def select_folder():
#     global workpath
#     folder_source = QtWidgets.QFileDialog.getExistingDirectory(caption='Source folder with raw files')
#     print( folder_source )
#     workpath= os.path.join( folder_source,'krill_data' )
    
#     os.chdir( workpath )
    
#     if not os.path.isfile('settings.ini'):
#         settings_default(folder_source)
    
if os.path.isfile('settings.ini'):
    config = configparser.ConfigParser()
    config.read('settings.ini') 
else:
    settings_default(r'D:\2022807-ANTARCTIC-PROVIDER')
   
folder_source=  str(config['GENERAL']['source_folder'])
w.pass_source_folder(folder_source)   
w_email.pass_source_folder(folder_source)   

fn=os.path.basename(os.path.normpath(folder_source))



workpath=os.path.join( str( pathlib.Path().resolve() ), 'krill_data_' + fn)


if not os.path.isdir(workpath):
    os.mkdir(workpath)

w.pass_work_folder(workpath)   
w_email.pass_work_folder(workpath)   
 
shutil.copy2('settings.ini', workpath) # complete target filename given
os.chdir(workpath)

# add a button widget and configure with the call back
# button_select = Button(label="Select folder")
# button_select.on_click(select_folder)

toggle = Toggle(label="Start processing",button_type = "success")
def start_stop_processing(attr):
    if toggle.active:
        thread.started.connect(w.start)
        thread.start()
        toggle.label = "Stop processing"
        toggle.button_type = "danger"
    else:    
        # thread.quit()
        thread.terminate()        
        toggle.label = "Start processing"
        toggle.button_type = "success"
toggle.on_click(start_stop_processing)

toggle_email = Toggle(label="Send mails",button_type = "success")

def start_stop_email(attr):
    if toggle_email.active:
        thread_email.started.connect(w_email.start)
        thread_email.start()
        toggle_email.label = "Stop sending mails"
        toggle_email.button_type = "danger"
    else:    
        # thread_email.quit()
        thread_email.terminate()        
        toggle_email.label = "Send mails"
        toggle_email.button_type = "success"
toggle_email.on_click(start_stop_email)

toggle_interpolation = Toggle(label="Interpolate maps")

# def update_plots():
#  if toggle_realtimeplot.active:        # print('Update plots')
#         self.plottimer = QtCore.QTimer(self)
#         self.plottimer.timeout.connect(self.scan_and_vizualize)
#         self.plottimer.start(10000)  
#         # self.checkbox_manual.setChecked(False)           
#     else:
#         # print('STOP plots')
#         self.plottimer.stop()  
 

           
def scan_and_vizualize():
    # global source
        
    def dothis():
        config = configparser.ConfigParser()
        config.read('settings.ini') 
        sv_min=float(config['GRAPHICS']['sv_min'])
        sv_max=float(config['GRAPHICS']['sv_max'])
        
        echogram_time_length=30
                    
        os.chdir(workpath)
        nasc_done =  np.array( glob.glob( '*_nasctable.h5' ) )
        
        # print(nasc_done)
        
        if len(nasc_done)>0:
                        
            nascfile_times=pd.to_datetime( nasc_done,format='D%Y%m%d-T%H%M%S_nasctable.h5' )
            time_threshold=nascfile_times.max() +pd.to_timedelta(10,'min') - pd.to_timedelta( 5 , 'days' )
            nasc_load= nasc_done[ nascfile_times>time_threshold ]
            
            
            df_nasc=pd.DataFrame([])
            for file in nasc_load:
                df=pd.read_hdf(file,key='df')
                df_nasc=pd.concat([df_nasc,df  ])
            df_nasc=df_nasc.sort_values('ping_time')   
            
            timedelta =np.concatenate([np.array([0]), np.diff( df_nasc.index.values ) /1e9 ])           
            df_nasc['speed_ms'] = df_nasc['distance_m'] / timedelta.astype(float)
            df_nasc['speed_knots']=df_nasc['speed_ms']*1.94384449
            ix_too_slow = df_nasc['speed_knots'] < float(config['GRAPHICS']['min_speed_in_knots'])           
            df_nasc.loc[ix_too_slow,'nasc']=np.nan           
    
            df_krig=df_nasc.resample('10min').mean() 
            
            # self.startdate.setDateTime(QtCore.QDateTime.fromString( nascfile_times.min().strftime('%Y-%m-%d %X') , "yyyy-MM-dd HH:mm:ss") )
            # self.enddate.setDateTime(QtCore.QDateTime.fromString( nascfile_times.max().strftime('%Y-%m-%d %X') , "yyyy-MM-dd HH:mm:ss") )
    
            
            latlim=[ df_nasc['lat'].min() , df_nasc['lat'].max()  ]
            lonlim=[ df_nasc['lon'].min() , df_nasc['lon'].max()  ]
    
    
    
            # get onl the latest echograms
            sv_done = np.array( glob.glob( '*_sv_swarm.h5' ) )
            
            sv_times=pd.to_datetime( sv_done,format='D%Y%m%d-T%H%M%S_sv_swarm.h5' )
            time_threshold=sv_times.max()  - pd.to_timedelta( echogram_time_length , 'min' )
            sv_load= sv_done[ sv_times>time_threshold ]
            df_nasc_load= df_nasc[ df_nasc.index>time_threshold ]
            # distance_covered= geod.line_length(lons=df_nasc_load['lon'],lats=df_nasc_load['lat']) / 1000  
            distance_covered=df_nasc_load['distance_m'].sum() /1000         
            
            
            df_sv=pd.DataFrame([])
            for file in sv_load:
                df=pd.read_hdf(file,key='df')
                # df=df.resample('1min').mean()
                df_sv=pd.concat([df_sv,df])
                
            ix_latest = (df_sv.index.max() -  df_sv.index) <  pd.to_timedelta( echogram_time_length , 'min' )
            df_sv= df_sv.iloc[ix_latest,:]   
            
            pv = np.rot90( df_sv.values )
            pv[pv<sv_min]=sv_min
            pv[pv>sv_max]=sv_max
            
            # print(pv)
            
            source.data={'image':[pv] ,'x':[-distance_covered],'y':[-df_sv.columns[-1].astype(float)],'dw':[distance_covered],'dh':[df_sv.columns[-1].astype(float) ] } # OK
            
        
            
            lat=df_krig['lat']
            lon=df_krig['lon']
            
            lonlat_to_webmercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            x, y = lonlat_to_webmercator.transform(lon,lat)
            
            df_map=pd.DataFrame(columns=['x','y','z'])
            df_map['x']=x
            df_map['y']=y
            df_map['z']=df_krig['nasc'].values
            
            source_map.data=df_map
            source_map_last.data={'x':[df_map.iloc[-1,0]],'y':[df_map.iloc[-1,1]] }
        
            if toggle_interpolation.active:          
        
                    try:
                        # a =df_krig['nasc'].values.copy()
                        # a[a>nasc_cutoff]=np.nan
                        
                        ix= (df_krig['lat'].notna()) & (df_krig['nasc'].notna() )                        
        
                        
                        OK = OrdinaryKriging(
                            360+df_krig.loc[ix,'lon'].values,
                            df_krig.loc[ix,'lat'].values,
                            df_krig.loc[ix,'nasc'].values,
                            coordinates_type='geographic',
                            variogram_model="spherical",
                            # variogram_parameters = { 'range' :  np.rad2deg( 50 / 6378  ) ,'sill':700000,'nugget':600000}
                            )
                         
                        d_lats=np.linspace(latlim[0],latlim[1],500)
                        d_lons=np.linspace(lonlim[0],lonlim[1],500)
                        
                        lat,lon=np.meshgrid( d_lats,d_lons )
                        
                        z1, ss1 = OK.execute("grid", d_lons, d_lats)                                  
                        z_grid=z1.data
                        z_grid[ ss1.data>np.percentile( ss1.data,75) ] =np.nan
                        
                        x, y = lonlat_to_webmercator.transform(df_nasc['lon'].min(),df_nasc['lat'].min())
                        x2, y2 = lonlat_to_webmercator.transform(df_nasc['lon'].max(),df_nasc['lat'].max())
                        

                        source_map_grid.data={'image':[z_grid],'x':[x],'y':[y],'dw':[x2-x],'dh':[y2-y]}
                        # source_map_grid_meta.data={'x':[x],'y':[y],'dw':[x2-x],'dh':[y2-y]}
                        
                        # print(source_map_grid.data)
         
                    except Exception() as e:
                        print(e)       
            else:
                   source_map_grid.data={'image':[],'x':[],'y':[],'dw':[],'dh':[]} 
    doc.add_next_tick_callback( dothis )      

    # if toggle_realtimeplot.active:                             
    #     plottimer = Timer(3, scan_and_vizualize)
    #     plottimer.start()

   


toggle_realtimeplot = Toggle(label="Start realtime plotting",button_type = "success")

# plottimer = QtCore.QTimer()
# plottimer.timeout.connect(scan_and_vizualize)
# print(plottimer)

# plottimer = Timer(3, scan_and_vizualize)

# class RepeatTimer(Timer):  
#     def run(self):  
#         while not self.finished.wait(self.interval):  
#             self.function(*self.args,**self.kwargs)  
            

def start_stop_live_plotting(attr):
    global callback_plots 
    if toggle_realtimeplot.active:
        
        callback_plots= doc.add_periodic_callback( scan_and_vizualize,3000 ) 
        
        
        toggle_realtimeplot.label = "Stop realtime plotting"
        toggle_realtimeplot.button_type = "danger"
    else:    
        # plottimer.cancel() 
        doc.remove_periodic_callback(callback_plots)
        
        toggle_realtimeplot.label = "Start realtime plotting"
        toggle_realtimeplot.button_type = "success"
toggle_realtimeplot.on_click(start_stop_live_plotting)
     


# settings_input = FileInput()
# label='Import settings (*.ini file)'

# button_edit_settings = Button(label="Edit settings")
# def edit_settings():
#      os.startfile('settings.ini')
# button_edit_settings.on_click( edit_settings   )


button_showfolder = Button(label="Show data folder")
def showfolder():
      os.startfile(workpath)
button_showfolder.on_click( showfolder   )




from bokeh.models import ColumnDataSource

# only modify from a Bokeh session callback
# source = ColumnDataSource(data=pd.DataFrame([]))
# z = np.random.randn(100,100)

source = ColumnDataSource({'image':[],'x':[],'y':[],'dw':[],'dh':[]}) # OK
doc = curdoc()

TOOLTIPS = [("x", "$x"),
    ("y", "$y"),
    ("value", "@image")]

plot1 = figure(tooltips=TOOLTIPS,active_drag="box_zoom", x_axis_label='Distance in km',y_axis_label='Depth in m')
plot1.sizing_mode = 'scale_both'


# config = configparser.ConfigParser()
# config.read('settings.ini') 
# sv_min=float(config['GRAPHICS']['sv_min'])
# sv_max=float(config['GRAPHICS']['sv_max'])

color_mapper = LinearColorMapper(palette="Viridis256")
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12,title='s_V in dB')
plot1.add_layout(color_bar, 'right')


plot1.image('image',source=source,x='x',y='y',dw='dw',dh='dh',color_mapper=color_mapper,level="image")


# plot1.image('image',source=source,x=0,y=0,dw=10,dh=10,palette='Inferno256') # OK


# lat=np.random.randn(100)*-60
# lon=np.random.randn(100)*-40
# nasc=np.random.randn(100)

# from pyproj import Transformer
# # TRAN_4326_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857")
# # x,y= TRAN_4326_TO_3857.transform(df['lon'], df['lat'])
# lonlat_to_webmercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
# x, y = lonlat_to_webmercator.transform(lon,lat)

df_map=pd.DataFrame(columns=['x','y','z'])
# df_map['x']=x
# df_map['y']=y
# df_map['z']=nasc

source_map = ColumnDataSource(data=df_map) # OK
source_map_last = ColumnDataSource(data= {'x':[],'y':[] }) # OK

# 
# df=pd.read_hdf('example_nasctable.h5',key='df')

# df=pd.DataFrame([])

TOOLTIPS_map = [("x", "$x"),
    ("y", "$y"),
    ("value", "$z")]

plot2 = figure(x_axis_type="mercator", y_axis_type="mercator", tooltips=TOOLTIPS_map,active_drag="box_zoom")

# plot2 = figure(x_range=(-9000000  ,-3000000 ), y_range=(-9000000, -7000000),
#            x_axis_type="mercator", y_axis_type="mercator", tooltips=TOOLTIPS_map,active_drag="box_zoom")

# tile_provider = get_provider(CARTODBPOSITRON)
# plot2.add_tile(tile_provider)

tilpro = get_provider(CARTODBPOSITRON)
plot2.add_tile(tilpro)


plot2.sizing_mode = 'scale_both'

color_mapper_map = LinearColorMapper(palette="Viridis256", low=0)
color_bar = ColorBar(color_mapper=color_mapper_map, label_standoff=12,title='NASC')
plot2.add_layout(color_bar, 'right')


plot2.circle('x','y',source=source_map, size=20, alpha=0.9 ,
              line_color=None, fill_color={"field":"z", "transform":color_mapper_map}) 



plot2.triangle('x','y',source=source_map_last, size=20, line_color=None, fill_color='red') 


# source_map_grid=ColumnDataSource(data={'image':[z]})
# source_map_grid_meta=ColumnDataSource(data={'x':[0],'y':[0],'dw':[1000],'dh':[1000]})

source_map_grid=ColumnDataSource(data={'image':[],'x':[],'y':[],'dw':[],'dh':[]})



# print( source_map_grid_meta.data['x'][0] )

# x=source_map_grid_meta.data['x'][0]
# y=source_map_grid_meta.data['y'][0]
# dw=source_map_grid_meta.data['dw'][0]
# dh=source_map_grid_meta.data['dh'][0]

# plot2.image('image',source=source_map_grid,x=x,y=y,dw=dw,dh=dh,color_mapper=color_mapper_map,level="image")
plot2.image('image',source=source_map_grid,x='x',y='y',dw='dw',dh='dh',color_mapper=color_mapper_map,level="image")


curdoc().add_root( row( column(button_showfolder,toggle_email ,toggle,toggle_realtimeplot,toggle_interpolation), row(plot1,plot2)  ))
# curdoc().theme = 'dark_minimal'

