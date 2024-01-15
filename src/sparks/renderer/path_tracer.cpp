#include "sparks/renderer/path_tracer.h"

#include "sparks/util/util.h"
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace sparks {

PathTracer::PathTracer(const RendererSettings *render_settings,
                       const Scene *scene) {
  render_settings_ = render_settings;
  scene_ = scene;
}

bool PathTracer::RR(double p) const{
  static std::default_random_engine e(time(NULL));
  static std::uniform_real_distribution<double> u1(0, 1);
  double rnd = u1(e);
  if (rnd < p)
    return true;
  else
    return false;
}

PathTracer::SampledResult PathTracer::UniformSampling(glm::vec3 axis) const {
  const double pi = 3.1415926;
  glm::vec3 next_ray = glm::ballRand(1.0f);
  if (glm::dot(next_ray, axis) < 0)
    next_ray = -next_ray;
  SampledResult sample;
  sample.ray = next_ray;
  sample.pdf = 1.0 / (2.0 * pi);
  return sample;
}
PathTracer::SampledResult PathTracer::ConeSampling(glm::vec3 axis) const {
  const double pi = 3.1415926;
  glm::vec3 next_ray = glm::ballRand(1.0f);
  while (glm::dot(next_ray, axis) < 0.99) {
    next_ray = glm::ballRand(1.0f);
  }
  SampledResult sample;
  sample.ray = next_ray;
  sample.pdf = 1.0 / (0.02 * pi);
  return sample;
}

PathTracer::SampledResult PathTracer::CosWeightedSampling(glm::vec3 axis, float exp) const {
  const double pi = 3.1415926;
  std::default_random_engine e(time(NULL));
  std::uniform_real_distribution<double> u1(0, 1);
  std::uniform_real_distribution<double> u2(0, 1);
  float rnd1 = u1(e), rnd2 = u2(e);
  float phi = 2 * pi * rnd2;
  float theta = acos(pow(1 - rnd1, 1.0f / exp));
  glm::vec3 sup(axis.y, axis.z, axis.x);
  glm::vec3 vert = glm::normalize(glm::cross(axis, sup));

  glm::vec3 phiaxis = axis + sin(theta) / cos(theta) * vert;
  phiaxis = glm::normalize(phiaxis);
  glm::vec3 sampled = phiaxis * cos(phi) +
                      glm::cross(axis, phiaxis) * sin(phi) +
                      axis * glm::dot(axis, phiaxis) * (1 - cos(phi));
  SampledResult sample;
  sample.pdf = sin(theta) * pow(cos(theta), exp) * (exp + 1) / 2.0f / pi;
  sample.ray = glm::normalize(sampled);
  return sample;
}

PathTracer::SampledResult PathTracer::BRDFWeightedSampling(
    glm::vec3 axis,glm::vec3 wo, float alpha ) const {
  const double pi = 3.1415926;
  const float m_e = 2.7182818;
  std::default_random_engine e(time(NULL));
  std::uniform_real_distribution<double> u1(0, 1);
  std::uniform_real_distribution<double> u2(0, 1);
  float rnd1 = u1(e), rnd2 = u2(e);
  float phi = 2 * pi * rnd2;
  float theta = atan(sqrt(-alpha * alpha * log(1 - rnd1)));
  glm::vec3 sup(axis.y, axis.z, axis.x);
  glm::vec3 vert = glm::normalize(glm::cross(axis, sup));

  glm::vec3 phiaxis = axis + sin(theta) / cos(theta) * vert;
  phiaxis = glm::normalize(phiaxis);
  glm::vec3 wh = phiaxis * cos(phi) +
                      glm::cross(axis, phiaxis) * sin(phi) +
                      axis * glm::dot(axis, phiaxis) * (1 - cos(phi));
  glm::vec3 wi = 2.0f * glm::dot(wh,wo) * wh - wo;
  SampledResult sample;
  sample.pdf = 2 * sin(theta) / (pow(alpha,2) * pow(cos(theta), 3)) * pow(m_e, pow(tan(theta),2) / pow(alpha, 2)) / (4 * glm::dot(wo, wh));
  sample.ray = glm::normalize(wi);
  return sample;
}

float PathTracer::Phong_spec(glm::vec3 v1, glm::vec3 v2, glm::vec3 normal, float exp) const{
  glm::vec3 bivec = glm::normalize(v1 + v2);
  float cos_bn = glm::dot(bivec, normal);
  if (cos_bn > 0)
    return pow(cos_bn, exp);
  return 0.0;
}

glm::vec3 PathTracer::Shade(HitRecord intersection, glm::vec3 wo, int depth) const {
  const float pi = 3.1415926;
  auto &material = scene_->GetEntity(intersection.hit_entity_id).GetMaterial();
  float alpha = material.alpha;
  auto brdf_d =
      material.albedo_color *
      glm::vec3{scene_->GetTextures()[material.albedo_texture_id].Sample(
          intersection.tex_coord)} / pi;
  auto brdf_s =
      material.specular_color *
      glm::vec3{scene_->GetTextures()[material.albedo_texture_id].Sample(
          intersection.tex_coord)} / pi;
  glm::vec3 L_dir(0, 0, 0), L_ind(0, 0, 0);
  float sum_pdf = 0;

  // light source
  if (material.material_type == MATERIAL_TYPE_EMISSION) {
    return material.emission * material.emission_strength * glm::vec3 {
      scene_->GetTextures()[material.albedo_texture_id].Sample(
          intersection.tex_coord)
    };
  }

  // direct light
  // emission light
  static std::default_random_engine e(std::random_device{}());
  int num_ob = scene_->GetEntityCount();
  for (int i = 0; i < num_ob; i++) {
    auto &object = scene_->GetEntity(i);
    auto &material_i = scene_->GetEntity(i).GetMaterial();
    if (material_i.material_type == MATERIAL_TYPE_EMISSION) {
      glm::vec3 L_this(0.0, 0.0, 0.0);
      double total_area = 0;
      glm::vec3 xl, vn;
      const Mesh *mesh_ptr = reinterpret_cast<const Mesh *>(object.GetModel());
      auto &vertices = mesh_ptr->Get_vertices();
      auto &indices = mesh_ptr->Get_indices();
      int n_triangle_face = indices.size() / 3;
      double *triangle_area = new double[n_triangle_face];
      for (int j = 0; j < n_triangle_face; j++) {
        total_area +=
            reinterpret_cast<const Mesh *>(object.GetModel())->area(j);
        triangle_area[j] = total_area;
      }
      std::uniform_real_distribution<double> u1(0, total_area);
      double rnd = u1(e);
      for (int j = 0; j < n_triangle_face; j++) {
        if (rnd < triangle_area[j] && (j == 0 || rnd >= triangle_area[j - 1])) {
          auto v1 = vertices[indices[3 * j]], v2 = vertices[indices[3 * j + 1]],
               v3 = vertices[indices[3 * j + 2]];

          static std::uniform_real_distribution<double> u2(0, 1);
          double rnd1 = u2(e), rnd2 = u2(e), rnd3 = u2(e);
          float p1 = rnd1 / (rnd1 + rnd2 + rnd3),
                p2 = rnd2 / (rnd1 + rnd2 + rnd3),
                p3 = rnd3 / (rnd1 + rnd2 + rnd3);
          xl = v1.position * p1 + v2.position * p2 + v3.position * p3;
          vn = v1.normal * p1 + v2.normal * p2 + v3.normal * p3;
          break;
        }
      }
      delete[] triangle_area;
      glm::vec3 wi = glm::normalize(xl - intersection.position);
      float visible = 1;
      HitRecord hit_light;
      auto vt =
          scene_->TraceRay(intersection.position, wi, 1e-3f, 1e4f, &hit_light);
      auto va = glm::length(xl - intersection.position);
      if (hit_light.hit_entity_id != i && abs(va - vt) > 1e-3f)
        visible = 0;
      intersection.geometry_normal = glm::normalize(intersection.geometry_normal);
      auto dots = glm::dot(intersection.geometry_normal, wi);
      float pdf_light = float(1) / total_area;
      float cos_theta = abs(glm::dot(wi, hit_light.geometry_normal));
      float cos_theta_hat = abs(glm::dot(wi, intersection.geometry_normal));
      float dist = glm::distance(xl, intersection.position);
      float dist2 = dist * dist;
      glm::vec3 intensity =
          material_i.emission * material_i.emission_strength * glm::vec3{scene_->GetTextures()[material_i.albedo_texture_id].Sample(hit_light.tex_coord)};
      if (dots > 0) {
        float phong_spe = Phong_spec(wi,wo,intersection.geometry_normal, material.specular_exponent);
        if (material.material_type == MATERIAL_TYPE_LAMBERTIAN || material.material_type == MATERIAL_TYPE_GLOSSY)
          L_this += alpha * intensity * brdf_d * cos_theta * cos_theta_hat / dist2 / pdf_light * visible;
        if (material.material_type == MATERIAL_TYPE_GLOSSY)
          L_this += alpha * intensity * brdf_s * phong_spe  * cos_theta_hat / dist2 /
                   pdf_light * visible;
        if (material.material_type == MATERIAL_TYPE_TRANSMISSIVE)
          L_this += alpha * intensity * brdf_d * cos_theta * cos_theta_hat /
                   dist2 / pdf_light * visible;
      }
      if (dots < 0 && material.material_type == MATERIAL_TYPE_TRANSMISSIVE) {
        L_this += (1 - alpha) * intensity * brdf_d * cos_theta * cos_theta_hat /
                 dist2 / pdf_light * visible;
      }
      L_dir += L_this;
    }
  }

  // env light
  auto light_direction = scene_->GetEnvmapLightDirection();
  auto t_l = scene_->TraceRay(intersection.position, light_direction, 1e-3f,
                              1e4f, nullptr);
  if (t_l < 0.0f) {
    if (material.material_type == MATERIAL_TYPE_GLOSSY || material.material_type == MATERIAL_TYPE_LAMBERTIAN)
      L_dir +=
         2.0f * pi * scene_->GetEnvmapMajorColor() *
        std::max(glm::dot(light_direction, intersection.geometry_normal),
                 0.0f) * brdf_d;
    if (material.material_type == MATERIAL_TYPE_GLOSSY) {
      
      L_dir += 2.0f * pi * scene_->GetEnvmapMajorColor() *
               brdf_s *
               Phong_spec(light_direction, wo, intersection.geometry_normal,
                          material.specular_exponent);
    }
  }

  // indirect light
  float p_rr = 0.6;
  if (RR(p_rr) && depth <= 8) {
    SampledResult sampleray ;
    SampledResult sampleray2;
    float mis_sum = 0.0f;
    glm::vec3 reflect = glm::normalize(2.0f * glm::dot(intersection.geometry_normal, wo) *
                            intersection.geometry_normal - wo);
    if (material.material_type == MATERIAL_TYPE_GLOSSY) {
      //sampleray2 =
          //CosWeightedSampling(2.0f * glm::dot(intersection.geometry_normal, wo) * intersection.geometry_normal - wo, 400.0f);      
      sampleray2 = UniformSampling(intersection.geometry_normal);
      sampleray = CosWeightedSampling(reflect, 400.0f);

    }
    if (material.material_type == MATERIAL_TYPE_LAMBERTIAN) {
      sampleray = UniformSampling(intersection.geometry_normal);
    }
    if (material.material_type == MATERIAL_TYPE_TRANSMISSIVE) {
      std::default_random_engine e(time(NULL));
      std::uniform_real_distribution<double> u(0, 1);
      if (u(e) < alpha)
        sampleray = UniformSampling(intersection.geometry_normal);
      else
        sampleray = CosWeightedSampling(-intersection.geometry_normal, 1.0f);
    }
    if (material.material_type == MATERIAL_TYPE_SPECULAR) {
      sampleray.ray = reflect;
      sampleray.pdf = 1.0;
    }
    glm::vec3 next_ray = sampleray.ray;
    float pdf = sampleray.pdf;
    HitRecord next_intersection;
    auto t_n = scene_->TraceRay(intersection.position, next_ray, 1e-3f, 1e4f,
                              &next_intersection);
    if (t_n > 0) {
      auto hitmaterial_type = scene_->GetEntity(next_intersection.hit_entity_id)
                                  .GetMaterial()
                                  .material_type;
      glm::vec3 intensity =
          Shade(next_intersection, -next_ray, depth + 1) / p_rr;
      if (material.material_type != MATERIAL_TYPE_SPECULAR &&
          hitmaterial_type != MATERIAL_TYPE_EMISSION) {
          L_ind += brdf_d * intensity *
                   abs(glm::dot(intersection.geometry_normal, next_ray)) / pdf;
    }
      if (material.material_type == MATERIAL_TYPE_SPECULAR)
        L_ind += intensity;
      if (material.material_type == MATERIAL_TYPE_GLOSSY)
        L_ind += intensity * brdf_s *
                 Phong_spec(next_ray, wo, intersection.geometry_normal,
                            material.specular_exponent) /
                 float(sampleray.pdf);

    } else if (t_n < 0) {
      if (material.material_type != MATERIAL_TYPE_SPECULAR)
        L_ind += brdf_d * glm::vec3{scene_->SampleEnvmap(next_ray)} *
               abs(glm::dot(intersection.geometry_normal, next_ray)) / pdf;
      if (material.material_type == MATERIAL_TYPE_SPECULAR)
        L_ind += glm::vec3{scene_->SampleEnvmap(next_ray)};
      if (material.material_type == MATERIAL_TYPE_GLOSSY) {
        L_ind += brdf_s * glm::vec3{scene_->SampleEnvmap(sampleray.ray)} *
                 Phong_spec(next_ray, wo, intersection.geometry_normal,
                            material.specular_exponent) /
                 float(sampleray.pdf);
      }
    }
    if (material.material_type == MATERIAL_TYPE_GLOSSY) {
      mis_sum += pdf;
      L_ind *= pdf;
      auto t_n = scene_->TraceRay(intersection.position, sampleray2.ray, 1e-3f, 1e4f,
                                  &next_intersection);
      if (t_n > 0) {
        glm::vec3 intensity =
            Shade(next_intersection, -sampleray2.ray, depth + 1) / p_rr;
          L_ind += intensity * brdf_s *
                   Phong_spec(sampleray2.ray, wo, intersection.geometry_normal,
                              material.specular_exponent);
          L_ind += brdf_d * intensity *
                   abs(glm::dot(intersection.geometry_normal, sampleray2.ray));
      } else {
        L_ind += brdf_d * glm::vec3{scene_->SampleEnvmap(sampleray2.ray)} *
                 abs(glm::dot(intersection.geometry_normal, sampleray2.ray));
        L_ind += brdf_s * glm::vec3{scene_->SampleEnvmap(sampleray2.ray)} *
                 Phong_spec(sampleray2.ray, wo, intersection.geometry_normal,
                            material.specular_exponent);
      }
      mis_sum += sampleray2.pdf;
      L_ind /= mis_sum;
    }
        
  }

  return L_dir + L_ind;
}

glm::vec3 PathTracer::SampleRay(glm::vec3 origin,
                                glm::vec3 direction,
                                int x,
                                int y,
                                int sample) const {
  glm::vec3 radiance(0, 0, 0);
  HitRecord intersection;
  auto t = scene_->TraceRay(origin, direction, 1e-3f, 1e4f, &intersection);
  if (t > 0) {
    radiance = sparks::PathTracer::Shade(intersection, -direction, 0);
  } else {
    radiance = glm::vec3{scene_->SampleEnvmap(direction)};
  }
  return radiance;
}
}  // namespace sparks
