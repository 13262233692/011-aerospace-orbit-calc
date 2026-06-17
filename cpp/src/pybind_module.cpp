#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include "sgp4_model.hpp"
#include "sgp4_propagator.hpp"
#include "wgs84_converter.hpp"
#include "batch_propagator.hpp"
#include "perturbation_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_sgp4_binding, m) {
    m.doc() = "SGP4 orbit propagator C++ binding for LEO satellite trajectory calculation";

    py::class_<sgp4::ElsetRec>(m, "ElsetRec")
        .def(py::init<>())
        .def_readwrite("satnum", &sgp4::ElsetRec::satnum)
        .def_readwrite("epochyr", &sgp4::ElsetRec::epochyr)
        .def_readwrite("epochdays", &sgp4::ElsetRec::epochdays)
        .def_readwrite("ndot", &sgp4::ElsetRec::ndot)
        .def_readwrite("nddot", &sgp4::ElsetRec::nddot)
        .def_readwrite("bstar", &sgp4::ElsetRec::bstar)
        .def_readwrite("inclo", &sgp4::ElsetRec::inclo)
        .def_readwrite("nodeo", &sgp4::ElsetRec::nodeo)
        .def_readwrite("ecco", &sgp4::ElsetRec::ecco)
        .def_readwrite("argpo", &sgp4::ElsetRec::argpo)
        .def_readwrite("mo", &sgp4::ElsetRec::mo)
        .def_readwrite("no_kozai", &sgp4::ElsetRec::no_kozai)
        .def_readwrite("no_unkozai", &sgp4::ElsetRec::no_unkozai)
        .def_readwrite("a", &sgp4::ElsetRec::a)
        .def_readwrite("satname", &sgp4::ElsetRec::satname)
        .def_readwrite("error", &sgp4::ElsetRec::error);

    py::class_<sgp4::Vec3>(m, "Vec3")
        .def(py::init<>())
        .def_readwrite("x", &sgp4::Vec3::x)
        .def_readwrite("y", &sgp4::Vec3::y)
        .def_readwrite("z", &sgp4::Vec3::z);

    py::class_<sgp4::PropagationResult>(m, "PropagationResult")
        .def(py::init<>())
        .def_readwrite("position", &sgp4::PropagationResult::position)
        .def_readwrite("velocity", &sgp4::PropagationResult::velocity);

    py::class_<sgp4::TimedPosition>(m, "TimedPosition")
        .def(py::init<>())
        .def_readwrite("timestamp_epoch", &sgp4::TimedPosition::timestamp_epoch)
        .def_readwrite("x", &sgp4::TimedPosition::x)
        .def_readwrite("y", &sgp4::TimedPosition::y)
        .def_readwrite("z", &sgp4::TimedPosition::z)
        .def_readwrite("vx", &sgp4::TimedPosition::vx)
        .def_readwrite("vy", &sgp4::TimedPosition::vy)
        .def_readwrite("vz", &sgp4::TimedPosition::vz);

    py::class_<sgp4::BatchResult>(m, "BatchResult")
        .def(py::init<>())
        .def_readwrite("satellite_id", &sgp4::BatchResult::satellite_id)
        .def_readwrite("epoch_jd", &sgp4::BatchResult::epoch_jd)
        .def_readwrite("positions", &sgp4::BatchResult::positions)
        .def("to_numpy", [](const sgp4::BatchResult& br) {
            size_t n = br.positions.size();
            py::array_t<double> timestamps(n);
            py::array_t<double> xs(n);
            py::array_t<double> ys(n);
            py::array_t<double> zs(n);
            py::array_t<double> vxs(n);
            py::array_t<double> vys(n);
            py::array_t<double> vzs(n);

            auto ts_buf = timestamps.mutable_unchecked<1>();
            auto x_buf = xs.mutable_unchecked<1>();
            auto y_buf = ys.mutable_unchecked<1>();
            auto z_buf = zs.mutable_unchecked<1>();
            auto vx_buf = vxs.mutable_unchecked<1>();
            auto vy_buf = vys.mutable_unchecked<1>();
            auto vz_buf = vzs.mutable_unchecked<1>();

            for (size_t i = 0; i < n; i++) {
                ts_buf(i) = br.positions[i].timestamp_epoch;
                x_buf(i) = br.positions[i].x;
                y_buf(i) = br.positions[i].y;
                z_buf(i) = br.positions[i].z;
                vx_buf(i) = br.positions[i].vx;
                vy_buf(i) = br.positions[i].vy;
                vz_buf(i) = br.positions[i].vz;
            }

            py::dict result;
            result["timestamps"] = timestamps;
            result["x"] = xs;
            result["y"] = ys;
            result["z"] = zs;
            result["vx"] = vxs;
            result["vy"] = vys;
            result["vz"] = vzs;
            return result;
        });

    m.def("parse_tle", &sgp4::parse_tle,
          py::arg("line1"), py::arg("line2"),
          "Parse TLE two-line element set");

    m.def("parse_tle_with_name", &sgp4::parse_tle_with_name,
          py::arg("name"), py::arg("line1"), py::arg("line2"),
          "Parse TLE with satellite name");

    m.def("sgp4_init", &sgp4::sgp4_init,
          py::arg("rec"),
          "Initialize SGP4 model from TLE elements");

    m.def("sgp4_propagate", &sgp4::sgp4_propagate,
          py::arg("rec"), py::arg("tsince"),
          "Propagate SGP4 orbit for tsince minutes");

    m.def("eci_to_wgs84", [](double x, double y, double z,
                              double vx, double vy, double vz, double jd) {
        sgp4::ECIPosition eci;
        eci.x = x; eci.y = y; eci.z = z;
        eci.vx = vx; eci.vy = vy; eci.vz = vz;
        sgp4::WGS84Position wgs84 = sgp4::eci_to_wgs84(eci, jd);
        py::dict result;
        result["x"] = wgs84.x;
        result["y"] = wgs84.y;
        result["z"] = wgs84.z;
        result["vx"] = wgs84.vx;
        result["vy"] = wgs84.vy;
        result["vz"] = wgs84.vz;
        return result;
    }, py::arg("x"), py::arg("y"), py::arg("z"),
       py::arg("vx"), py::arg("vy"), py::arg("vz"), py::arg("jd"),
       "Convert ECI coordinates to WGS84 ECEF");

    py::class_<sgp4::BatchECEFBuffer, std::shared_ptr<sgp4::BatchECEFBuffer>>(
        m, "BatchECEFBuffer",
        "Batch ECEF coordinate buffer with shared memory management. "
        "Numpy views reference the same underlying C++ memory, which is "
        "automatically freed when all references are released.")
        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("size"))
        .def_property_readonly("size", &sgp4::BatchECEFBuffer::size)
        .def_property_readonly("empty", &sgp4::BatchECEFBuffer::empty)
        .def("clear", &sgp4::BatchECEFBuffer::clear, "Release all internal memory")
        .def("resize", &sgp4::BatchECEFBuffer::resize, py::arg("size"),
             "Resize the buffer (invalidates existing views)")
        .def("to_numpy_dict", [](std::shared_ptr<sgp4::BatchECEFBuffer> self) {
            size_t n = self->size();
            py::dict result;
            auto make_view = [&](const char* name, double* data) {
                py::array_t<double> arr(
                    py::array::ShapeContainer{n},
                    py::array::StridesContainer{sizeof(double)},
                    data,
                    py::cast(self)
                );
                result[name] = arr;
            };
            make_view("x", self->x());
            make_view("y", self->y());
            make_view("z", self->z());
            make_view("vx", self->vx());
            make_view("vy", self->vy());
            make_view("vz", self->vz());
            return result;
        }, "Return dict of numpy array views (zero-copy, base=this buffer)")
        .def("get_x", [](std::shared_ptr<sgp4::BatchECEFBuffer> self) {
            return py::array_t<double>(
                py::array::ShapeContainer{self->size()},
                py::array::StridesContainer{sizeof(double)},
                self->x(),
                py::cast(self)
            );
        }, "Zero-copy numpy view of X coordinates")
        .def("get_y", [](std::shared_ptr<sgp4::BatchECEFBuffer> self) {
            return py::array_t<double>(
                py::array::ShapeContainer{self->size()},
                py::array::StridesContainer{sizeof(double)},
                self->y(),
                py::cast(self)
            );
        }, "Zero-copy numpy view of Y coordinates")
        .def("get_z", [](std::shared_ptr<sgp4::BatchECEFBuffer> self) {
            return py::array_t<double>(
                py::array::ShapeContainer{self->size()},
                py::array::StridesContainer{sizeof(double)},
                self->z(),
                py::cast(self)
            );
        }, "Zero-copy numpy view of Z coordinates")
        .def("get_vx", [](std::shared_ptr<sgp4::BatchECEFBuffer> self) {
            return py::array_t<double>(
                py::array::ShapeContainer{self->size()},
                py::array::StridesContainer{sizeof(double)},
                self->vx(),
                py::cast(self)
            );
        }, "Zero-copy numpy view of X velocity")
        .def("get_vy", [](std::shared_ptr<sgp4::BatchECEFBuffer> self) {
            return py::array_t<double>(
                py::array::ShapeContainer{self->size()},
                py::array::StridesContainer{sizeof(double)},
                self->vy(),
                py::cast(self)
            );
        }, "Zero-copy numpy view of Y velocity")
        .def("get_vz", [](std::shared_ptr<sgp4::BatchECEFBuffer> self) {
            return py::array_t<double>(
                py::array::ShapeContainer{self->size()},
                py::array::StridesContainer{sizeof(double)},
                self->vz(),
                py::cast(self)
            );
        }, "Zero-copy numpy view of Z velocity")
        .def("use_count", [](std::shared_ptr<sgp4::BatchECEFBuffer>& self) {
            return self.use_count();
        }, "Reference count of the underlying shared buffer (for debugging)");

    m.def("eci_to_ecef_buffer", [](py::array_t<double, py::array::c_style | py::array::forcecast> eci_x_arr,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> eci_y_arr,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> eci_z_arr,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> eci_vx_arr,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> eci_vy_arr,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> eci_vz_arr,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> jd_arr) {
        auto x_buf = eci_x_arr.unchecked<1>();
        auto y_buf = eci_y_arr.unchecked<1>();
        auto z_buf = eci_z_arr.unchecked<1>();
        auto vx_buf = eci_vx_arr.unchecked<1>();
        auto vy_buf = eci_vy_arr.unchecked<1>();
        auto vz_buf = eci_vz_arr.unchecked<1>();
        auto jd_buf = jd_arr.unchecked<1>();

        size_t n = x_buf.shape(0);

        return sgp4::eci_to_ecef_buffer(
            x_buf.data(0), y_buf.data(0), z_buf.data(0),
            vx_buf.data(0), vy_buf.data(0), vz_buf.data(0),
            jd_buf.data(0), n
        );
    }, py::arg("eci_x"), py::arg("eci_y"), py::arg("eci_z"),
       py::arg("eci_vx"), py::arg("eci_vy"), py::arg("eci_vz"), py::arg("jd_array"),
       "Batch convert ECI to ECEF and return a shared BatchECEFBuffer (zero-copy views)");

    m.def("eci_to_ecef_batch_cpp", [](py::array_t<double, py::array::c_style | py::array::forcecast> eci_x_arr,
                                       py::array_t<double, py::array::c_style | py::array::forcecast> eci_y_arr,
                                       py::array_t<double, py::array::c_style | py::array::forcecast> eci_z_arr,
                                       py::array_t<double, py::array::c_style | py::array::forcecast> eci_vx_arr,
                                       py::array_t<double, py::array::c_style | py::array::forcecast> eci_vy_arr,
                                       py::array_t<double, py::array::c_style | py::array::forcecast> eci_vz_arr,
                                       py::array_t<double, py::array::c_style | py::array::forcecast> jd_arr) {
        auto x_buf = eci_x_arr.unchecked<1>();
        auto y_buf = eci_y_arr.unchecked<1>();
        auto z_buf = eci_z_arr.unchecked<1>();
        auto vx_buf = eci_vx_arr.unchecked<1>();
        auto vy_buf = eci_vy_arr.unchecked<1>();
        auto vz_buf = eci_vz_arr.unchecked<1>();
        auto jd_buf = jd_arr.unchecked<1>();

        size_t n = x_buf.shape(0);

        sgp4::BatchECEFResult res = sgp4::eci_to_ecef_batch(
            x_buf.data(0), y_buf.data(0), z_buf.data(0),
            vx_buf.data(0), vy_buf.data(0), vz_buf.data(0),
            jd_buf.data(0), n
        );

        py::dict result;
        result["x"] = py::array_t<double>(n, res.x.data());
        result["y"] = py::array_t<double>(n, res.y.data());
        result["z"] = py::array_t<double>(n, res.z.data());
        result["vx"] = py::array_t<double>(n, res.vx.data());
        result["vy"] = py::array_t<double>(n, res.vy.data());
        result["vz"] = py::array_t<double>(n, res.vz.data());
        return result;
    }, py::arg("eci_x"), py::arg("eci_y"), py::arg("eci_z"),
       py::arg("eci_vx"), py::arg("eci_vy"), py::arg("eci_vz"), py::arg("jd_array"),
       "Batch convert ECI coordinates to WGS84 ECEF using C++ acceleration");

    m.def("propagate_7days", &sgp4::propagate_7days,
          py::arg("rec"), py::arg("step_seconds") = 1,
          "Propagate satellite orbit for 7 days at given step interval");

    py::class_<sgp4::LockFreePerturbationEngine>(
        m, "LockFreePerturbationEngine",
        "Lock-free perturbation engine for atmospheric drag and solar radiation pressure. "
        "Uses double-buffering with memory barriers to safely share real-time space "
        "environment parameters between update threads and computation threads without locks.")
        .def(py::init<>())
        .def_property_readonly_static("PARAM_COUNT", [](py::object) {
            return sgp4::LockFreePerturbationEngine::PARAM_COUNT;
        })
        .def("update_param", &sgp4::LockFreePerturbationEngine::update_param,
             py::arg("index"), py::arg("value"),
             "Update single parameter (thread-safe, lock-free)")
        .def("update_all", [](sgp4::LockFreePerturbationEngine& self,
                               py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
            auto buf = arr.unchecked<1>();
            if (buf.shape(0) != sgp4::LockFreePerturbationEngine::PARAM_COUNT) {
                throw std::runtime_error("Array must have " +
                    std::to_string(sgp4::LockFreePerturbationEngine::PARAM_COUNT) + " elements");
            }
            self.update_all(buf.data(0));
        }, py::arg("params"),
           "Update all parameters from numpy array (thread-safe, lock-free)")
        .def("get_param", &sgp4::LockFreePerturbationEngine::get_param,
             py::arg("index"),
             "Get single parameter value (thread-safe)")
        .def("get_all", [](const sgp4::LockFreePerturbationEngine& self) {
            py::array_t<double> arr(sgp4::LockFreePerturbationEngine::PARAM_COUNT);
            auto buf = arr.mutable_unchecked<1>();
            self.get_all(buf.mutable_data(0));
            return arr;
        }, "Get all parameters as numpy array (thread-safe)")
        .def("sequence", &sgp4::LockFreePerturbationEngine::sequence,
             "Current update sequence number")
        .def("has_update", &sgp4::LockFreePerturbationEngine::has_update,
             py::arg("last_seq"),
             "Check if new data is available since last sequence number")
        .def("apply_perturbation", [](sgp4::LockFreePerturbationEngine& self,
                                        py::array_t<double, py::array::c_style> x_arr,
                                        py::array_t<double, py::array::c_style> y_arr,
                                        py::array_t<double, py::array::c_style> z_arr,
                                        py::array_t<double, py::array::c_style> vx_arr,
                                        py::array_t<double, py::array::c_style> vy_arr,
                                        py::array_t<double, py::array::c_style> vz_arr,
                                        double jd, double dt) {
            auto x = x_arr.mutable_unchecked<1>();
            auto y = y_arr.mutable_unchecked<1>();
            auto z = z_arr.mutable_unchecked<1>();
            auto vx = vx_arr.mutable_unchecked<1>();
            auto vy = vy_arr.mutable_unchecked<1>();
            auto vz = vz_arr.mutable_unchecked<1>();
            if (x.shape(0) != 1 || y.shape(0) != 1 || z.shape(0) != 1 ||
                vx.shape(0) != 1 || vy.shape(0) != 1 || vz.shape(0) != 1) {
                throw std::runtime_error("All arrays must have exactly 1 element");
            }
            self.apply_perturbation(
                x.mutable_data(0), y.mutable_data(0), z.mutable_data(0),
                vx.mutable_data(0), vy.mutable_data(0), vz.mutable_data(0),
                jd, dt
            );
        }, py::arg("x"), py::arg("y"), py::arg("z"),
           py::arg("vx"), py::arg("vy"), py::arg("vz"),
           py::arg("jd"), py::arg("dt"),
           "Apply perturbation correction in-place using 1-element numpy arrays")
        .def("apply_perturbation_single", [](const sgp4::LockFreePerturbationEngine& self,
                                             double x, double y, double z,
                                             double vx, double vy, double vz,
                                             double jd, double dt) -> py::tuple {
            double xo = x, yo = y, zo = z;
            double vxo = vx, vyo = vy, vzo = vz;
            self.apply_perturbation(&xo, &yo, &zo, &vxo, &vyo, &vzo, jd, dt);
            return py::make_tuple(xo, yo, zo, vxo, vyo, vzo);
        }, py::arg("x"), py::arg("y"), py::arg("z"),
           py::arg("vx"), py::arg("vy"), py::arg("vz"),
           py::arg("jd"), py::arg("dt"),
           "Apply perturbation correction and return new values as tuple")
        .def("apply_perturbation_batch", [](const sgp4::LockFreePerturbationEngine& self,
                                             py::array_t<double> x_arr,
                                             py::array_t<double> y_arr,
                                             py::array_t<double> z_arr,
                                             py::array_t<double> vx_arr,
                                             py::array_t<double> vy_arr,
                                             py::array_t<double> vz_arr,
                                             py::array_t<double> jd_arr,
                                             double step_sec) {
            if (x_arr.ndim() != 1 || y_arr.ndim() != 1 || z_arr.ndim() != 1 ||
                vx_arr.ndim() != 1 || vy_arr.ndim() != 1 || vz_arr.ndim() != 1 ||
                jd_arr.ndim() != 1) {
                throw std::runtime_error("All arrays must be 1-dimensional");
            }
            py::ssize_t n_signed = x_arr.shape(0);
            if (y_arr.shape(0) != n_signed || z_arr.shape(0) != n_signed ||
                vx_arr.shape(0) != n_signed || vy_arr.shape(0) != n_signed ||
                vz_arr.shape(0) != n_signed || jd_arr.shape(0) != n_signed) {
                throw std::runtime_error("All arrays must have the same size");
            }
            size_t n = static_cast<size_t>(n_signed);
            if (n == 0) return;

            auto x_buf = x_arr.request();
            auto y_buf = y_arr.request();
            auto z_buf = z_arr.request();
            auto vx_buf = vx_arr.request();
            auto vy_buf = vy_arr.request();
            auto vz_buf = vz_arr.request();

            std::vector<double> x_copy, y_copy, z_copy, vx_copy, vy_copy, vz_copy;
            bool need_x = (x_buf.strides[0] != (py::ssize_t)sizeof(double));
            bool need_y = (y_buf.strides[0] != (py::ssize_t)sizeof(double));
            bool need_z = (z_buf.strides[0] != (py::ssize_t)sizeof(double));
            bool need_vx = (vx_buf.strides[0] != (py::ssize_t)sizeof(double));
            bool need_vy = (vy_buf.strides[0] != (py::ssize_t)sizeof(double));
            bool need_vz = (vz_buf.strides[0] != (py::ssize_t)sizeof(double));

            double* x_ptr = static_cast<double*>(x_buf.ptr);
            double* y_ptr = static_cast<double*>(y_buf.ptr);
            double* z_ptr = static_cast<double*>(z_buf.ptr);
            double* vx_ptr = static_cast<double*>(vx_buf.ptr);
            double* vy_ptr = static_cast<double*>(vy_buf.ptr);
            double* vz_ptr = static_cast<double*>(vz_buf.ptr);

            if (need_x) {
                x_copy.resize(n);
                double* src = x_ptr;
                for (size_t i = 0; i < n; ++i) { x_copy[i] = src[i * (x_buf.strides[0] / sizeof(double))]; }
                x_ptr = x_copy.data();
            }
            if (need_y) {
                y_copy.resize(n);
                double* src = y_ptr;
                for (size_t i = 0; i < n; ++i) { y_copy[i] = src[i * (y_buf.strides[0] / sizeof(double))]; }
                y_ptr = y_copy.data();
            }
            if (need_z) {
                z_copy.resize(n);
                double* src = z_ptr;
                for (size_t i = 0; i < n; ++i) { z_copy[i] = src[i * (z_buf.strides[0] / sizeof(double))]; }
                z_ptr = z_copy.data();
            }
            if (need_vx) {
                vx_copy.resize(n);
                double* src = vx_ptr;
                for (size_t i = 0; i < n; ++i) { vx_copy[i] = src[i * (vx_buf.strides[0] / sizeof(double))]; }
                vx_ptr = vx_copy.data();
            }
            if (need_vy) {
                vy_copy.resize(n);
                double* src = vy_ptr;
                for (size_t i = 0; i < n; ++i) { vy_copy[i] = src[i * (vy_buf.strides[0] / sizeof(double))]; }
                vy_ptr = vy_copy.data();
            }
            if (need_vz) {
                vz_copy.resize(n);
                double* src = vz_ptr;
                for (size_t i = 0; i < n; ++i) { vz_copy[i] = src[i * (vz_buf.strides[0] / sizeof(double))]; }
                vz_ptr = vz_copy.data();
            }

            auto jd_buf = jd_arr.request();
            auto jd_buf_s = jd_arr.strides(0);
            std::vector<double> jd_copy_storage;
            const double* jd_ptr = static_cast<const double*>(jd_buf.ptr);
            if (jd_buf_s != (py::ssize_t)sizeof(double)) {
                jd_copy_storage.resize(n);
                for (size_t i = 0; i < n; ++i) { jd_copy_storage[i] = jd_ptr[i * (jd_buf_s / sizeof(double))]; }
                jd_ptr = jd_copy_storage.data();
            }

            self.apply_perturbation_batch(x_ptr, y_ptr, z_ptr, vx_ptr, vy_ptr, vz_ptr, jd_ptr, step_sec, n);

            if (need_x) {
                double* dst = static_cast<double*>(x_buf.ptr);
                for (size_t i = 0; i < n; ++i) { dst[i * (x_buf.strides[0] / sizeof(double))] = x_copy[i]; }
            }
            if (need_y) {
                double* dst = static_cast<double*>(y_buf.ptr);
                for (size_t i = 0; i < n; ++i) { dst[i * (y_buf.strides[0] / sizeof(double))] = y_copy[i]; }
            }
            if (need_z) {
                double* dst = static_cast<double*>(z_buf.ptr);
                for (size_t i = 0; i < n; ++i) { dst[i * (z_buf.strides[0] / sizeof(double))] = z_copy[i]; }
            }
            if (need_vx) {
                double* dst = static_cast<double*>(vx_buf.ptr);
                for (size_t i = 0; i < n; ++i) { dst[i * (vx_buf.strides[0] / sizeof(double))] = vx_copy[i]; }
            }
            if (need_vy) {
                double* dst = static_cast<double*>(vy_buf.ptr);
                for (size_t i = 0; i < n; ++i) { dst[i * (vy_buf.strides[0] / sizeof(double))] = vy_copy[i]; }
            }
            if (need_vz) {
                double* dst = static_cast<double*>(vz_buf.ptr);
                for (size_t i = 0; i < n; ++i) { dst[i * (vz_buf.strides[0] / sizeof(double))] = vz_copy[i]; }
            }
        }, py::arg("x"), py::arg("y"), py::arg("z"),
           py::arg("vx"), py::arg("vy"), py::arg("vz"),
           py::arg("jd_array"), py::arg("step_sec"),
           "Apply perturbation correction to batch of coordinates in-place (handles non-contiguous arrays)");

    py::class_<sgp4::BatchPerturbationEngine,
               std::shared_ptr<sgp4::BatchPerturbationEngine>>(
        m, "BatchPerturbationEngine",
        "Multi-satellite perturbation engine managing multiple LockFreePerturbationEngine instances. "
        "Supports global environmental updates and per-satellite configuration.")
        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("num_satellites"))
        .def("__getitem__", [](sgp4::BatchPerturbationEngine& self, size_t index) -> sgp4::LockFreePerturbationEngine& {
            if (index >= self.size()) {
                throw py::index_error();
            }
            return self[index];
        }, py::return_value_policy::reference_internal,
           "Get satellite-specific perturbation engine by index")
        .def_property_readonly("size", &sgp4::BatchPerturbationEngine::size,
                               "Number of satellite engines")
        .def("resize", &sgp4::BatchPerturbationEngine::resize, py::arg("new_size"),
             "Resize the number of satellite engines")
        .def("update_global_env", &sgp4::BatchPerturbationEngine::update_global_env,
             py::arg("f10_7"), py::arg("f10_7_avg"), py::arg("kp"), py::arg("storm_level"),
             "Update global space environment for all satellites (thread-safe)");
}
