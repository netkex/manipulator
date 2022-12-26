from manipulator import * 
from obstacle import *
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from copy import deepcopy

eps = 1e-9

black = 'black'
red = px.colors.qualitative.Dark24[3]
grey = px.colors.qualitative.D3[7]
green = px.colors.qualitative.Alphabet[6]
purple = px.colors.qualitative.Alphabet[0]
blue = px.colors.qualitative.Dark24[19]
cyan = px.colors.qualitative.Alphabet[18]

init_sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Step:",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"easing": "cubic-in-out"},
    "pad": {"b": 15, "t": 30},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": []
}
camera_pose = np.ones(3)


def animate(
    manipulators: List[Manipulator], 
    obstacles: List[Obstacle],
    goal_point: np.ndarray,
    goal_size: float,
    x_range: List[float] = None,
    y_range: List[float] = None,
    z_range: List[float] = None, 
    init_camera_pose: np.ndarray | List[float] = None,
    fps: float = 30,
    width: int = 800, 
    height: int = 700,
    joint_width: int = 3,
    line_width: int = 2,
    manipulator_color: str = black,
    manipulator_bottom_color: str = None,
    title: str = '',
    show_grid: bool = True,
    shadow_coeff: float = 0.95, 
    draw_ground: bool = False, 
    ground_color: str = black,
    format: str = 'show',
    path_to_html: str = 'manipulator.html',
    auto_open: bool = True):  
    """
    Manipulator motion animation

    Parameters
    ----------
    manipulators : List[Manipulator]
        Manipualator states trace
    obstacles : List[Obstacle]
        Map obstacles
    goal_point : np.ndarray
    goal_size : float 
    x_range : List[float], optional 
        Box range on the x axis
    y_range : List[float], optional 
        Box range on the y axis
    z_range : List[float], optional 
        Box range on the z axis
    init_camera_pose : np.ndarray | List[float], optional, default [1, 1, 1]
        Relative initial camera position
    fps : float, optional, default 30 
        Manipulator states per second 
    width : int, optional, default 800
        Plot width
    height : int, optional, default 700
        Plot height
    joint_width : int, optional, default 3
        Joint marker width 
    line_width : int, optional, default 2
        Joint line width 
    manipulator_color : str, optional, default black
    manipulator_bottom_color : str, optional, default manipulator_color
    title : str, optional, default ''
        Plot title 
    show_grid : bool, optional, default True
    shadow_coeff : float, optional, default 0.95
    draw_ground : bool, optional, default False
    ground_color : bool, optional, default black

    format : str, optional, default 'show'
        Type of output format 
        * 'show' - show HTML inplace 
        * 'html' - save HTML in file @path_to_html 
    path_to_html : str, optinal, default 'manipulator.html'
        Path to HTML file
    auto_open : bool, optional, default True
    """

    if x_range is None: 
        x_range = get_range(manipulators, 0) 
    if y_range is None: 
        y_range = get_range(manipulators, 1) 
    if z_range is None: 
        z_range = get_range(manipulators, 2)
    if manipulator_bottom_color is None: 
        manipulator_bottom_color = manipulator_color

    objects = [] 
    objects.append(create_sphere(goal_point, goal_size, green, 1.))
    for obstacle in obstacles: 
        if type(obstacle) == SphereObstacle: 
            objects.append(create_sphere(obstacle.center, obstacle.r, red, shadow_coeff))
        else: 
            raise RuntimeError('not implemented')
    if draw_ground: 
        objects.append(create_ground(x_range, y_range, 0, ground_color))

    sliders_dict = deepcopy(init_sliders_dict)
    fig_dict = {
        'frames': [],
        'layout': build_layout(
            x_range=x_range, 
            y_range=y_range, 
            z_range=z_range,
            init_camera_pose=init_camera_pose,
            width=width,
            height=height,
            fps=fps,
            title=title,
            show_grid=show_grid,
        )
    }

    for step, manipulator in enumerate(manipulators): 
        coords = manipulator.get_joint_coordinates()
        manipulator_plot = go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            marker=dict(
                size=joint_width,
                color=[manipulator_bottom_color] + (manipulator.joint_num - 1) * [manipulator_color],
            ),
            line=dict(
                width=line_width,
                color=manipulator_color,
            )
        )
        frame = dict(data=manipulator_plot, name=str(step))
        if step == 0: 
            fig_dict['data'] = [manipulator_plot] +  objects
        fig_dict['frames'].append(frame)
        slider_step = {
            "args": [[step], {"frame": {"redraw": True}, "mode": "immediate"}],
            "label": str(step),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]
    fig = go.Figure(fig_dict)

    if format == 'show': 
        fig.show() 
    elif format == 'html': 
        pio.write_html(fig, file=path_to_html, auto_open=auto_open)
    else: 
        raise RuntimeError('unknown format type')
    return

def get_range(
    manipulators: List[Manipulator],
    axis: int
) -> List[int]: 
    coords = list(map(lambda manipulator: manipulator.get_joint_coordinates(), manipulators))
    min_bound = int(min(map(lambda joints: np.min(joints[:, axis]), coords)) - 1 + eps)
    max_bound = int(max(map(lambda joints: np.max(joints[:, axis]), coords)) + 1 - eps) 
    return [min_bound, max_bound]    


def build_layout(
    x_range: List[float],
    y_range: List[float],
    z_range: List[float],
    init_camera_pose: np.ndarray = None,
    width: int = 800, 
    height: int = 700,
    fps: float = 30,
    title: str = '',
    show_grid: bool = True,
) -> go.Layout: 
    if init_camera_pose is None: 
        init_camera_pose = camera_pose

    prop_x = x_range[1] - x_range[0]
    prop_y = y_range[1] - y_range[0]
    prop_z = z_range[1] - z_range[0]
    prop_sum = prop_x + prop_y + prop_z
    prop_x /= prop_sum
    prop_y /= prop_sum
    prop_z /= prop_sum

    if show_grid: 
        scene = dict(
            xaxis=dict(title='x', range=x_range),
            yaxis=dict(title='y', range=y_range),
            zaxis=dict(title='z', range=z_range),
            aspectmode='manual',
            aspectratio=dict(
                x=prop_x, 
                y=prop_y, 
                z=prop_z
            ),
        )
    else:
        scene = dict(
            xaxis=dict(
                range=x_range, 
                titlefont_color='white', 
                backgroundcolor='white',
                color='white',
                gridcolor='white',
                showgrid=False,
                showline=False,
            ),
            yaxis=dict(
                range=y_range, 
                titlefont_color='white', 
                backgroundcolor='white',
                color='white',
                gridcolor='white',
                showgrid=False,
                showline=False,
            ),
            zaxis=dict(
                range=z_range, 
                titlefont_color='white', 
                backgroundcolor='white',
                color='white',
                gridcolor='white',
                showgrid=False,
                showline=False,
            ),
            aspectmode='manual',
            aspectratio=dict(
                x=prop_x, 
                y=prop_y, 
                z=prop_z
            ),
        )

    return go.Layout(
        title=title,
        width=width,
        height=height,
        hoverdistance=-1,
        hovermode='closest',
        scene=scene,
        scene_camera=dict(
            eye=dict(
                x = init_camera_pose[0] * prop_x, 
                y = init_camera_pose[1] * prop_y, 
                z = init_camera_pose[2] * prop_z
            ),
        ),
        margin=dict(r=5, l=5, b=5, t=30),
        updatemenus=[{'buttons':[
                {
                    "args": [None, {"frame": {"duration": int(1000 / fps), "redraw": True},
                                    "fromcurrent": True, "transition": {"easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            'direction': 'left',
            "pad": {"r": 10, "t": 85},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }])


def create_ground(
    x_range: List[float], 
    y_range: List[float],
    z: float = 0,
    color: str = black,
) -> go.Surface: 
    x = [x_range[0], x_range[0], x_range[1], x_range[1]]
    y = [y_range[0], y_range[1], y_range[0], y_range[1]]
    z = [[z] * 4] * 4
    ground_surface = go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], lighting=dict(ambient=1))
    ground_surface.update(showscale=False)
    return ground_surface


def create_sphere(
    center: np.ndarray, 
    r: float,
    color: str,
    shadow_coeff: float = 0.95,
    surface_point_num: int = 100,
) -> go.Surface:
    theta = np.linspace(0, 2 * np.pi, surface_point_num)
    phi = np.linspace(0, np.pi, surface_point_num)
    x = center[0] + r * np.outer(np.cos(theta), np.sin(phi))
    y = center[1] + r * np.outer(np.sin(theta), np.sin(phi))
    z = center[2] + r * np.outer(np.ones(100), np.cos(phi))
    sphere_surface = go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], lighting=dict(ambient=shadow_coeff))
    sphere_surface.update(showscale=False)
    return sphere_surface