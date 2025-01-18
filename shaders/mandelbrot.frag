#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    float time;
    dvec2 resolution;
    dvec2 offset_px;
    double scale;
} pushConstants;

#define STEPS 256
#define cx_mul(a, b) dvec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)

vec3 getColorFromPalette(float t) {
    const int numColors = 7;

    vec3 colors[numColors];
    colors[0] = vec3(0.5, 0.0, 1.0);
    colors[1] = vec3(1.0, 0.5, 0.0);
    colors[2] = vec3(1.0, 1.0, 0.0);
    colors[3] = vec3(0.0, 1.0, 0.0);
    colors[4] = vec3(0.0, 0.0, 1.0);
    colors[5] = vec3(0.3, 0.0, 0.5);
    colors[6] = vec3(0.0, 0.0, 0.0);

    t = clamp(t, 0.0, 1.0);

    float segment = 1.0 / float(numColors - 1);
    int index = int(t / segment);
    index = clamp(index, 0, numColors - 2);
    float t2 = (t - float(index) * segment) / segment;

    return mix(colors[index], colors[index + 1], smoothstep(0.0, 1.0, t2));
}

int julia_f(dvec2 p, dvec2 c)
{
    for(int i = 0; i < STEPS; ++i)
    {
      p = cx_mul(p, p) + c;
      if(length(p) >= 2) {
        return i;
      }
    }
    return STEPS;
}

void main() {
    const float t = pushConstants.time;
    const double scale = pushConstants.scale;

    const dvec2 pos = dvec2(gl_FragCoord.x, gl_FragCoord.y) / min(pushConstants.resolution.x,pushConstants.resolution.y);
    const dvec2 offset = pushConstants.offset_px / min(pushConstants.resolution.x,pushConstants.resolution.y);
    const dvec2 center = -dvec2(1.0, 0.5);

    dvec2 uv = (pos + center)/scale + offset;

    float steps = float(julia_f(vec2(0, 0), uv)) / float(STEPS);
    //float steps = float(julia_f(uv, vec2(-1, 0))) / float(STEPS);

    outColor = vec4(getColorFromPalette(steps), 1.0);
}
