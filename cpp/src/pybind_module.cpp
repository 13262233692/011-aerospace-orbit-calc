#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include "sgp4_model.hpp"
#include "sgp4_propagator.hpp"
#include "wgs84_converter.hpp"
#include "batch_propagator.hpp"

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
}
