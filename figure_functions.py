
def adjust_axis(ax, x, y, w, h):
    pos = ax.get_position()
    ax.set_position([pos.x0+x, pos.y0+y, pos.width*w, pos.height*h])