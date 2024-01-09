#pragma once
#include "random"
#include "sparks/assets/scene.h"
#include "sparks/renderer/renderer_settings.h"

namespace sparks {
class PathTracer {
 public:
  PathTracer(const RendererSettings *render_settings, const Scene *scene);
  [[nodiscard]] glm::vec3 SampleRay(glm::vec3 origin,
                                    glm::vec3 direction,
                                    int x,
                                    int y,
                                    int sample) const;
  glm::vec3 Shade(HitRecord interseciton, glm::vec3 wo, int depth) const;
  glm::vec3 VolumetricShade(HitRecord intersection, glm::vec3 wo, int depth) const;
  bool RR(double p) const;
  float PathTracer::Phong_spec(glm::vec3 v1,
                               glm::vec3 v2,
                               glm::vec3 normal,
                               float exp) const;

 private:
  const RendererSettings *render_settings_{};
  const Scene *scene_{};
};
}  // namespace sparks
