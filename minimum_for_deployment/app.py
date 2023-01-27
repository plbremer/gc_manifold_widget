from json.tool import main
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import io
import base64
import matplotlib.image as mpimg

from rdkit.Chem import Draw

import dash
from dash import dcc, html, dash_table, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

nearby_threshold=0.1

local_stylesheet = {
    "href": "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap",
    "rel": "stylesheet"
}

dataset=pd.read_pickle('all_attributes_for_widget_minimal.bin')
#dataset=pd.read_pickle('../intermediates/tiny_test.bin')


app = dash.Dash(__name__, use_pages=False, external_stylesheets=[dbc.themes.BOOTSTRAP, local_stylesheet ])

app.layout=html.Div(
    children=[
        dbc.Row(
            children=[
                dbc.Col(
                    children=[
                        html.H4('Coordinates based on:',className='text-center'),
                        html.Div(className="radio-group-container add-margin-top-1", 
                            children=[
                                html.Div(className="radio-group", 
                                    children=[
                                        dbc.RadioItems(
                                            id='radioitems_coordinates',
                                            options=[
                                                {'label': 'spectrum', 'value': 'spectrum'},
                                                {'label': 'structure', 'value': 'structure'},
                                            ],  
                                            value='spectrum',
                                            #value='binvestigate,species,organ,disease',
                                            className="btn-group",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            inputCheckedClassName="active"                               
                                        ),
                                    ]
                                )
                            ]
                        ),
                        html.Br(),
                        html.H4('Colors based on:',className='text-center'),
                        html.Div(className="radio-group-container add-margin-top-1", 
                            children=[
                                html.Div(className="radio-group", 
                                    children=[
                                        dbc.RadioItems(
                                            id='radioitems_colors',
                                            options=[
                                                {'label': 'molecular_weight', 'value': 'molecular_weight'},
                                                {'label': 'kovats_ri', 'value': 'Kovats-RI-est'},
                                                {'label': 'spectrum', 'value': 'spectrum_cluster_label'},
                                                {'label': 'structure', 'value': 'structure_cluster_label'},
                                                {'label': 'none', 'value': None}
                                            ],         
                                            #value='binvestigate,species,organ,disease',
                                            value='spectrum_cluster_label',
                                            className="btn-group",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            inputCheckedClassName="active"                               
                                        ),
                                    ]
                                )
                            ]
                        ),
                        html.Br(),
                        html.H4('Opacity:',className='text-center'),
                        html.Div(className="radio-group-container add-margin-top-1", 
                            children=[
                                html.Div(className="radio-group", 
                                    children=[
                                        dbc.RadioItems(
                                            id='radioitems_opacity',
                                            options=[
                                                {'label': '1', 'value': 1},
                                                {'label': '0.1', 'value': 0.1},
                                                {'label': '0.01', 'value': 0.01},
                                                {'label': '0.001', 'value': 0.001},
                                            ],       
                                            value=1,  
                                            #value='binvestigate,species,organ,disease',
                                            className="btn-group",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            inputCheckedClassName="active"                               
                                        ),
                                    ]
                                )
                            ]
                        ),

                        html.Br(),
                        html.H4('Show Points that Failed to Cluster:',className='text-center'),
                        html.Div(className="radio-group-container add-margin-top-1", 
                            children=[
                                html.Div(className="radio-group", 
                                    children=[
                                        dbc.RadioItems(
                                            id='radioitems_show_unlabeled',
                                            options=[
                                                {'label': 'True', 'value': True},
                                                {'label': 'False', 'value': False},
                                                # {'label': '0.01', 'value': '0.01'},
                                                # {'label': '0.001', 'value': '0.001'},
                                            ],         
                                            value=True,
                                            className="btn-group",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            inputCheckedClassName="active"                               
                                        ),
                                    ]
                                )
                            ]
                        ),

                        html.Br(),
                        html.Div(
                            dbc.Button(
                                'Draw Graph',
                                id='button_query',
                            ),
                            className="d-grid gap-2 col-2 mx-auto",
                        ),                                                  
                    ]
                )
            ]
        ),
        html.Br(),
        html.Br(),
        dbc.Row(
            children=[
                dbc.Col(
                    children=[
                        dbc.Spinner(
                            children=[
                                html.Div(
                                    id='div_manifold',
                                    children=[],
                                    #className="sunburst-container"
                                )
                            ]
                        ),
                    ],
                    width=6
                ),
                dbc.Col(
                    children=[
                        dbc.Spinner(
                            children=[
                                html.Div(
                                    id='div_compound_selection',
                                    children=[],
                                    #className="sunburst-container"
                                )
                            ]
                        ),                         
                    ],
                    width=6
                )                
            ]
        )
    ]
)


@callback(
    [
        Output(component_id='div_manifold', component_property='children'),
    ],
    [
        Input(component_id='button_query', component_property='n_clicks'),
    ],
    [
        State(component_id='radioitems_coordinates',component_property='value'),
        State(component_id='radioitems_colors',component_property='value'),
        State(component_id='radioitems_opacity',component_property='value'),
        State(component_id='radioitems_show_unlabeled',component_property='value'),
    ],
    prevent_initial_call=True
)
def generate_manifold_projection(button_query_n_clicks,radioitems_coordinates_value,radioitems_colors_value,radioitems_opacity_value,radioitems_show_unlabeled_value):

    temp_dataset=dataset

    #sort out coordinates
    if radioitems_coordinates_value=='spectrum':
        umap_x='umap_1_spectra'
        umap_y='umap_2_spectra'
        umap_z='umap_3_spectra'
    elif radioitems_coordinates_value=='structure':
        umap_x='umap_1_structure'
        umap_y='umap_2_structure'
        umap_z='umap_3_structure'
    #sort out color
    umap_color=radioitems_colors_value


    #sort out -1 cluster values
    if umap_color=='structure_cluster_label' or umap_color=='spectrum_cluster_label':
        if radioitems_show_unlabeled_value==False:
            temp_dataset=temp_dataset.loc[
                temp_dataset[umap_color]!='-1'
            ]


    manifold_projection_figure=px.scatter_3d(
        # other_coordinates.loc[
        #     (other_coordinates['Kovats-RI-est']<4000) &
        #     (other_coordinates['Kovats-RI-est']>1000)
        # ],
        temp_dataset,
        x=umap_x,
        y=umap_y,
        z=umap_z,
        #cold do scroll bar for opacity
        opacity=radioitems_opacity_value,
        color=umap_color
    )
                                                # {'label': 'molecular_weight', 'value': 'molecular_weight'},
                                                # {'label': 'kovats_ri', 'value': 'Kovats-RI-est'},
                                                # {'label': 'spectrum', 'value': 'spectrum_cluster_label'},
                                                # {'label': 'structure', 'value': 'structure_cluster_label'},

    if umap_color=='molecular_weight':
        manifold_projection_figure.update_layout(coloraxis=dict(
            cmin=49,
            cmax=700
        ))
    elif umap_color=='Kovats-RI-est':
        manifold_projection_figure.update_layout(coloraxis=dict(
            cmin=0,
            cmax=5000
        ))

    # fig.update_layout(coloraxis_colorbar=dict(
    #     title="Number of Bills per Cell",
    #     thicknessmode="pixels", thickness=50,
    #     lenmode="pixels", len=200,
    #     yanchor="top", y=1,
    #     ticks="outside", ticksuffix=" bills",
    #     dtick=5
    # ))



    div_manifold_children=[
        dbc.Row(
            children=[
                #dbc.Col(width={'size':2}),
                dbc.Col(
                    children=[
                        html.H2("Manifold Projection", className='text-center'),
                        html.H4("Click a point to see a random selection of nearby points", className='text-center'),
                        dcc.Graph(
                            id='manifold_figure',
                            figure= manifold_projection_figure,
                            style={
                                'height':800
                            }
                        ),
                    ],
                    #width={'size':8}
                ),
                #dbc.Col(width={'size':2}),
            ],
        ),        
    ]    

    return [div_manifold_children]


@callback(
    [
        Output(component_id='div_compound_selection', component_property='children'),
    ],
    [
        Input(component_id='manifold_figure', component_property='clickData'),
    ],
    [
        State(component_id='radioitems_coordinates',component_property='value'),
    #     State(component_id='radioitems_colors',component_property='value'),
    #     State(component_id='radioitems_opacity',component_property='value'),
    #     State(component_id='radioitems_show_unlabeled',component_property='value'),
    ],
    prevent_initial_call=True
)
def probe_point(manifold_figure_clickData,radioitems_coordinates_value):
    print(manifold_figure_clickData)

    point_x=manifold_figure_clickData['points'][0]['x']
    point_y=manifold_figure_clickData['points'][0]['y']
    point_z=manifold_figure_clickData['points'][0]['z']

    #sort out coordinates
    if radioitems_coordinates_value=='spectrum':
        umap_x='umap_1_spectra'
        umap_y='umap_2_spectra'
        umap_z='umap_3_spectra'
    elif radioitems_coordinates_value=='structure':
        umap_x='umap_1_structure'
        umap_y='umap_2_structure'
        umap_z='umap_3_structure'

    nearby_datapoints_panda=dataset.loc[
        (dataset[umap_x]<(point_x+nearby_threshold)) &
        (dataset[umap_x]>(point_x-nearby_threshold)) &
        (dataset[umap_y]<(point_y+nearby_threshold)) &
        (dataset[umap_y]>(point_y-nearby_threshold)) &
        (dataset[umap_z]<(point_z+nearby_threshold)) &
        (dataset[umap_z]>(point_z-nearby_threshold))            
    ]    

    ten_sample_rows=nearby_datapoints_panda.sample(
        n=10,
        replace=False,
        #random_state=1337
    )

    temp_image=draw_molecules_and_spectra(ten_sample_rows)

    print(nearby_datapoints_panda)

    #print(nearby_datapoints_panda.inchikey_first_block.value_counts())

    div_compound_selection_children=[
        html.Img(
            id='image_compounds',
            #height=1000,
            #width=1000,
            src=temp_image
        ),
    ]

    return [div_compound_selection_children]



def draw_molecules_and_spectra(sample_rows):
    fig,ax=plt.subplots(len(sample_rows.index),2,figsize=(10,20))

    sample_rows=sample_rows.reset_index(drop=True)

    for index,series in sample_rows.iterrows():
        #image_list.append(Draw.MolToImage(series['computed_rdkit_mol']))
        temp_mol_image=Draw.MolToImage(series['computed_rdkit_mol'])
        print(temp_mol_image)
        #temp_mol_image_numpy=mpimg.imread(temp_mol_image)
        ax[index,0].imshow(temp_mol_image)


        #
        #[[3.0, 1.0], [4.0, 0.8118118118118118], [5.0, ...
        ax[index,1].stem(
            #the added np.array is a hacky way to fix the axis limits
            np.append(series['spectrum_np'][:,0],np.array([0,500])),
            np.append(series['spectrum_np'][:,1],np.array([0,0])),
            markerfmt=" "
        )




    #plt.show()
    buf = io.BytesIO() # in-memory files
    plt.savefig(buf, format = "png") # save to the above file object
    plt.close('all')
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    plotly_fig="data:image/png;base64,{}".format(data)
    buf.close()
    return plotly_fig



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
    #app.run(debug=True)