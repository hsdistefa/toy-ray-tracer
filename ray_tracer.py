import numpy as np
import matplotlib.pyplot as plt


def normalize(vector):
    return vector / np.linalg.norm(vector)


def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center)**2 - radius**2
    delta = b**2 - 4*c
    if delta > 0:
        p1 = (-b + np.sqrt(delta)) / 2
        p2 = (-b - np.sqrt(delta)) / 2
        if p1 > 0 and p2 > 0:
            closest_point = min(p1, p2)
            return closest_point
        return None


def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for i, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[i]
    return nearest_object, min_distance


def render():
    width = 300
    height = 200

    camera = np.array([0, 0, 1])
    light = {'position': np.array([5, 5, 5]),
             'ambient': np.array([1, 1, 1]),
             'diffuse': np.array([1, 1, 1]),
             'specular': np.array([1, 1, 1])
             }
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    objects = [
        {'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]),
         'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100},
        {'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]),
         'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100},
        {'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]),
         'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100}
    ]

    image = np.zeros((height, width, 3))
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            print('progress: %d/%d' % (i + 1, height))
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)

            # Check for intersections
            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                continue

            # Compute intersection point between ray and nearest object
            intersection = origin + min_distance * direction

            # Shift the point perpendicular to the object surface to prevent
            # intersecting the object itself again
            normal_to_surface = normalize(intersection - nearest_object['center'])
            shifted_point = intersection + 1e-5 * normal_to_surface

            # Direction from object intersection to light source
            intersection_to_light = normalize(light['position'] - shifted_point)

            _, min_distance = nearest_intersected_object(objects, intersection, intersection_to_light)
            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed:
                continue

            # Render object using Blinn-Phong model

            # RGB
            illumination = np.zeros((3))

            # Ambient
            illumination += nearest_object['ambient'] * light['ambient']

            # Diffuse
            illumination += nearest_object['diffuse'] * light['diffuse'] * \
                np.dot(intersection_to_light, normal_to_surface)

            # Specular
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object['specular'] * light['specular'] * \
                np.dot(normal_to_surface, H) ** (nearest_object['shininess'] /4)

            image[i, j] = np.clip(illumination, 0, 1)

    plt.imsave('image.png', image)


if __name__ == '__main__':
    render()
