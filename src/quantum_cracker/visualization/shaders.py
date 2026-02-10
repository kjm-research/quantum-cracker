"""GLSL shader sources for the Quantum Cracker 3D renderer."""

VOXEL_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in float amplitude;
layout (location = 2) in float energy;
layout (location = 3) in float theta;
layout (location = 4) in float phi;

uniform mat4 view;
uniform mat4 projection;
uniform float time;

out float v_energy;
out float v_amplitude;

void main() {
    // Apply 78 MHz resonance vibration (the Compiler Effect)
    float vibration = sin(78.0 * phi + time) * cos(78.0 * theta);
    vec3 normal = normalize(position);
    vec3 displaced = position + normal * vibration * 0.02;

    gl_Position = projection * view * vec4(displaced, 1.0);
    gl_PointSize = max(1.0, energy * 10.0);
    v_energy = energy;
    v_amplitude = amplitude;
}
"""

VOXEL_FRAGMENT_SHADER = """
#version 330 core
in float v_energy;
in float v_amplitude;
out vec4 FragColor;

void main() {
    // Blue-to-red colormap based on energy
    float t = clamp(v_energy, 0.0, 1.0);
    vec3 color = mix(vec3(0.1, 0.2, 0.8), vec3(0.9, 0.1, 0.1), t);

    // Fade by amplitude
    float alpha = clamp(abs(v_amplitude) * 2.0, 0.1, 1.0);
    FragColor = vec4(color, alpha);
}
"""

THREAD_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in float visible;

uniform mat4 view;
uniform mat4 projection;

out float v_visible;

void main() {
    gl_Position = projection * view * vec4(position, 1.0);
    v_visible = visible;
}
"""

THREAD_FRAGMENT_SHADER = """
#version 330 core
in float v_visible;
out vec4 FragColor;

void main() {
    // Green if visible, red if not
    vec3 color = mix(vec3(0.9, 0.1, 0.1), vec3(0.1, 0.9, 0.2), v_visible);
    FragColor = vec4(color, 0.8);
}
"""
