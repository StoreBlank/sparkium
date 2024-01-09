#include "sparks/renderer/path_tracer.h"

#include "sparks/util/util.h"
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <cmath>
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

float PathTracer::Phong_spec(glm::vec3 v1, glm::vec3 v2, glm::vec3 normal, float exp) const{
  glm::vec3 bivec = glm::normalize(v1 + v2);
  float cos_bn = glm::dot(bivec, normal);
  if (cos_bn > 0)
    return pow(cos_bn, exp);
  return 0.0;
}

glm::vec3 PathTracer::VolumetricShade(HitRecord intersection,
                                      glm::vec3 wo,
                                      int depth) const {
  const float pi = 3.1415926;
  static std::default_random_engine e(std::random_device{}());
  // need to be adjusted according to the scene size
  const float ray_march_step_size = 0.3f;
  static std::uniform_real_distribution<double> u_jitter(0, ray_march_step_size);
  // jitter
  const float march_beginning = u_jitter(e);
  glm::vec3 shade_position = intersection.position;
  glm::vec3 direction = -glm::normalize(wo);
  auto &material = scene_->GetEntity(intersection.hit_entity_id).GetMaterial();
  float extinction = material.extinction;
  float scattering = material.scattering;
  float absorption = extinction - scattering;
  glm::vec3 emission = material.emission;
  float emission_strength = material.emission_strength;
  glm::vec3 L_e(0, 0, 0), L_dir(0, 0, 0), L_ind(0, 0, 0);

  HitRecord hit_self;
  auto length = scene_->TraceRay(shade_position, direction, 1e-2f, 1e4f, &hit_self);
  // too thin
  if (hit_self.hit_entity_id != intersection.hit_entity_id) {
    if (length < 0.0f)
      return glm::vec3{scene_->SampleEnvmap(direction)};
    else
      return Shade(hit_self, -direction, depth);
  }

  // emission light
  L_e += absorption / extinction * emission * emission_strength * (1 - exp(-extinction * length));

  // direct light, ray marching
  for (float distance = march_beginning; distance < length;
       distance += ray_march_step_size) {
    glm::vec3 pos = shade_position + direction * distance;
    float transmittance = exp(-distance * extinction);

    // enumerate light source
    int num_ob = scene_->GetEntityCount();
    for (int i = 0; i < num_ob; i++) {
      auto &object = scene_->GetEntity(i);
      auto &material_i = scene_->GetEntity(i).GetMaterial();
      if (material_i.material_type == MATERIAL_TYPE_EMISSION) {
        double total_area = 0;
        glm::vec3 xl, vn;
        const Mesh *mesh_ptr =
            reinterpret_cast<const Mesh *>(object.GetModel());
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
          if (rnd < triangle_area[j] &&
              (j == 0 || rnd >= triangle_area[j - 1])) {
            auto v1 = vertices[indices[3 * j]],
                 v2 = vertices[indices[3 * j + 1]],
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
        glm::vec3 light_direction = glm::normalize(xl - pos);
        float visible = 1;
        float light_dist_expected = glm::length(xl - pos);
        HitRecord hit_self_dir;
        float dir_transmittance = 1.0f;
        float dir_length = scene_->TraceRay(pos, light_direction, 1e-3f, 1e4f, &hit_self_dir);
        // too thin, check hit light
        if (hit_self_dir.hit_entity_id != intersection.hit_entity_id) {
          if (hit_self_dir.hit_entity_id != i ||
              abs(dir_length - light_dist_expected) > 1e-3f) {
            visible = 0;
          }
          float dots = 1.0f;
          float pdf_light = float(1) / total_area;
          float cos_theta =
              abs(glm::dot(light_direction, hit_self_dir.geometry_normal));
          float cos_theta_hat = 1.0f;
          float dist = light_dist_expected;
          float dist2 = dist * dist;
          glm::vec3 intensity =
              material_i.emission * material_i.emission_strength *
              glm::vec3{
                  scene_->GetTextures()[material_i.albedo_texture_id].Sample(
                      hit_self_dir.tex_coord)};
          L_dir += transmittance * intensity * cos_theta * cos_theta_hat /
                   dist2 / pdf_light * visible * scattering * 1.0f / (4 * pi) * ray_march_step_size;
        } else {
          dir_transmittance = exp(-dir_length * extinction);
          HitRecord hit_light;
          float light_to_surface = scene_->TraceRay(hit_self_dir.position, light_direction, 1e-2f, 1e4f, &hit_light);
          float light_dist = dir_length + light_to_surface;
          if (hit_light.hit_entity_id != i || abs(light_dist - light_dist_expected) > 1e-3f) {
						visible = 0;
					}
					float dots = 1.0f;
					float pdf_light = float(1) / total_area;
					float cos_theta =
							abs(glm::dot(light_direction, hit_light.geometry_normal));
					float cos_theta_hat = 1.0f;
					float dist = light_dist_expected;
					float dist2 = dist * dist;
					glm::vec3 intensity =
							material_i.emission * material_i.emission_strength *
            glm::vec3{
                  scene_->GetTextures()[material_i.albedo_texture_id].Sample(
											hit_self_dir.tex_coord)};
					L_dir += transmittance * intensity * cos_theta * cos_theta_hat /
									 dist2 / pdf_light * dir_transmittance * visible * scattering * 1.0f / (4 * pi) * ray_march_step_size;
        }
      }
    }

    // env light
    auto env_light_direction = scene_->GetEnvmapLightDirection();
    glm::vec3 env_light_intensity =
        2.0f * pi * scene_->GetEnvmapMajorColor();
    int entity_id = -1;
    float env_dir_transmittance = 1.0f;
    HitRecord hit_self_env;
    auto env_light_length = scene_->TraceRay(pos, env_light_direction, 1e-3f,
                                1e4f, &hit_self_env);
    entity_id = hit_self_env.hit_entity_id;
    if (entity_id == intersection.hit_entity_id) {
      env_dir_transmittance = exp(-env_light_length * extinction);
      HitRecord hit_env;
      scene_->TraceRay(hit_self_env.position, env_light_direction, 1e-2f, 1e4f, &hit_env);
      entity_id = hit_env.hit_entity_id;
    }
    if (entity_id == -1) {
      L_dir += transmittance * env_light_intensity * env_dir_transmittance * scattering * 1.0f / (4 * pi) * ray_march_step_size;
    }
  }

  // indirect light
  HitRecord hit_indirect;
  float transmittance = exp(-length * extinction);
  float dist_ind = scene_->TraceRay(hit_self.position, direction, 1e-3f, 1e4f, &hit_indirect);
  if (dist_ind < 0.0f) {
		L_ind += transmittance * glm::vec3{scene_->SampleEnvmap(direction)};
  }
  else {
		L_ind += transmittance * Shade(hit_indirect, -direction, depth);
	}

  return L_e + L_dir + L_ind;
}

glm::vec3 PathTracer::Shade(HitRecord intersection, glm::vec3 wo, int depth) const {
  const float pi = 3.1415926;
  auto &material = scene_->GetEntity(intersection.hit_entity_id).GetMaterial();
  if (material.material_type == MATERIAL_TYPE_VOLUMETRIC)
    return VolumetricShade(intersection, wo, depth);
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
      if (hit_light.hit_entity_id != i || abs(va - vt) > 1e-3f)
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
      if (dots > 0 && (material.material_type == MATERIAL_TYPE_GLOSSY || material.material_type == MATERIAL_TYPE_LAMBERTIAN)) {
        float phong_spe = Phong_spec(wi,wo,intersection.geometry_normal, material.specular_exponent);
        if (material.material_type != MATERIAL_TYPE_SPECULAR)
          L_dir += alpha * intensity * brdf_d * cos_theta * cos_theta_hat / dist2 / pdf_light * visible;
        if (material.material_type == MATERIAL_TYPE_GLOSSY ||
            material.material_type == MATERIAL_TYPE_SPECULAR)
          L_dir += alpha * intensity * brdf_s * phong_spe  * cos_theta_hat / dist2 /
                   pdf_light * visible;
      }
      if (dots < 0 && material.material_type == MATERIAL_TYPE_TRANSMISSIVE) {
        L_dir += (1 - alpha) * intensity * brdf_d * cos_theta * cos_theta_hat /
                 dist2 / pdf_light * visible;
      }
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
    if (material.material_type == MATERIAL_TYPE_GLOSSY ||
        material.material_type == MATERIAL_TYPE_SPECULAR) {
      
      L_dir += 2.0f * pi * scene_->GetEnvmapMajorColor() *
               brdf_s *
               Phong_spec(light_direction, wo, intersection.geometry_normal,
                          material.specular_exponent);
    }
  }

  // indirect light
  float p_rr = 0.6;
  if (RR(p_rr) && depth <= 8) {
    glm::vec3 next_ray = glm::ballRand(1.0f);
    if (glm::dot(next_ray, intersection.geometry_normal) < 0)
      next_ray = -next_ray;
    HitRecord next_intersection;
    auto t_n = scene_->TraceRay(intersection.position, next_ray, 1e-3f, 1e4f,
                              &next_intersection);
    if (t_n > 0 && scene_->GetEntity(next_intersection.hit_entity_id)
                           .GetMaterial()
                           .material_type != MATERIAL_TYPE_EMISSION) {
      glm::vec3 intensity = Shade(next_intersection, -next_ray, depth+1) / p_rr;
      if (material.material_type != MATERIAL_TYPE_SPECULAR)
        L_ind += alpha * 2.0f * pi * brdf_d * intensity *
               abs(glm::dot(intersection.geometry_normal, next_ray));
      if (material.material_type == MATERIAL_TYPE_GLOSSY ||
          material.material_type == MATERIAL_TYPE_SPECULAR) {
        L_ind +=
            alpha * 2.0f * pi * intensity * brdf_s *
            Phong_spec(next_ray, wo, intersection.geometry_normal,
                       material.specular_exponent);
      }

    } else if (t_n < 0) {
      if (material.material_type != MATERIAL_TYPE_SPECULAR)
        L_ind += alpha * 2.0f * pi * brdf_d * glm::vec3{scene_->SampleEnvmap(next_ray)} *
               abs(glm::dot(intersection.geometry_normal, next_ray));
      if (material.material_type == MATERIAL_TYPE_GLOSSY ||
          material.material_type == MATERIAL_TYPE_SPECULAR) {
        L_ind += alpha * 2.0f * pi * brdf_s *
                 glm::vec3{scene_->SampleEnvmap(next_ray)} *
                 Phong_spec(next_ray, wo, intersection.geometry_normal,
                            material.specular_exponent);
      }
    }
    if (material.material_type == MATERIAL_TYPE_TRANSMISSIVE) {
      next_ray = -next_ray;
      HitRecord next_intersection;
      auto t_n = scene_->TraceRay(intersection.position, next_ray, 1e-3f, 1e4f,
                                  &next_intersection);
      if (t_n > 0 && scene_->GetEntity(next_intersection.hit_entity_id)
                             .GetMaterial()
                             .material_type != MATERIAL_TYPE_EMISSION) {
        glm::vec3 intensity =
            Shade(next_intersection, -next_ray, depth + 1) / p_rr;
        L_ind += (1 - alpha) * 2.0f * pi * brdf_d * intensity *
                   abs(glm::dot(intersection.geometry_normal, next_ray));

      } else if (t_n < 0) {
        L_ind += (1 - alpha) * 2.0f * pi * brdf_d *
                   glm::vec3{scene_->SampleEnvmap(next_ray)} *
                   abs(glm::dot(intersection.geometry_normal, next_ray));
      }

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
