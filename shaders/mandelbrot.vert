#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform PushConstants {
    float time;
    dvec2 resolution;
    dvec2 offset_px;
    double scale;
} pushConstants;

void main() {
    gl_Position = vec4(inPosition.xy, 0.0, 1.0);
    fragColor = vec3(0.0, 0.0, 0.5);
}
