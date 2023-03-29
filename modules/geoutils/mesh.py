from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import Union

import geopandas as gpd
from shapely.geometry import Polygon

from .vector import get_intersect_polygons


# 地域メッシュの基底クラス
@dataclass
class RegionMesh(ABC):
    order: int

    def __post_init__(self) -> None:
        self.size_array = self.create_mesh_size_array(self.order)

    @staticmethod
    @abstractmethod
    def create_mesh_size_array(order: int) -> list[dict[str, Decimal]]:
        pass

    @abstractmethod
    def get_mesh_code(self, latitude: Union[float, Decimal], longitude: Union[float, Decimal]) -> int:
        pass

    @abstractmethod
    def calculate_mesh_point_from_code(mesh_code: int) -> tuple[Decimal]:
        pass

    def get_mesh_polygon_from_code(self, mesh_code: int) -> Polygon:
        x, y = self.calculate_mesh_point_from_code(mesh_code)
        return self.generate_polygon(x, y, self.size_array[self.order - 1])

    @staticmethod
    def generate_polygon(x: Decimal, y: Decimal, size: dict[str, Decimal]) -> Polygon:
        return Polygon([(x, y), (x, y + size['latitude']), (x + size['longitude'], y + size['latitude']),
                        (x + size['longitude'], y), (x, y)])

    def get_intersect_codes(self, bounds: tuple[float], mesh_size: dict[str, Decimal]) -> list[int]:
        codes = [self.get_mesh_code(lat, lon) for lat, lon in product([bounds[1], bounds[3]], [bounds[0], bounds[2]])]
        lat_list = range(int((Decimal(bounds[3]) - Decimal(bounds[1])) // mesh_size['latitude']) + 1)
        lon_list = range(int((Decimal(bounds[2]) - Decimal(bounds[0])) // mesh_size['longitude']) + 1)
        for idx_lat, idx_lon in product(lat_list, lon_list):
            latitude = Decimal(Decimal(bounds[1]) + mesh_size['latitude'] * Decimal(idx_lat))
            longitude = Decimal(Decimal(bounds[0]) + mesh_size['longitude'] * Decimal(idx_lon))
            codes.append(self.get_mesh_code(latitude, longitude))
        return sorted(list(set(codes)))

    def get_intersect_mesh(self, aoi: Polygon) -> gpd.GeoDataFrame:
        codes = self.get_intersect_codes(aoi.bounds, self.size_array[self.order - 1])
        gdf = gpd.GeoDataFrame([[code, self.get_mesh_polygon_from_code(code)] for code in codes],
                               columns=['code', 'geometry'])
        return get_intersect_polygons(gdf, aoi).set_crs(epsg=4326)


# 基準地域メッシュ
class StandardRegionMesh(RegionMesh):

    @staticmethod
    def create_mesh_size_array(order: int) -> list[dict[str, Decimal]]:
        size_array = [dict(latitude=Decimal(2. / 3.), longitude=Decimal(1.))]
        for i in range(1, order):
            coef = 8. if i == 1 else 10.
            size_array.append(
                dict(latitude=Decimal(size_array[-1]['latitude'] / Decimal(coef)),
                     longitude=Decimal(size_array[-1]['longitude'] / Decimal(coef))))
        return size_array

    def get_mesh_code(self, latitude: Union[float, Decimal], longitude: Union[float, Decimal]) -> int:
        code, _ = self.calculate_mesh_code_and_point(latitude, longitude)
        return int(code)

    def get_base_point(self, latitude: Union[float, Decimal], longitude: Union[float, Decimal]) -> dict[str, Decimal]:
        _, point = self.calculate_mesh_code_and_point(latitude, longitude)
        return point

    def calculate_mesh_code_and_point(self, latitude: Union[float, Decimal],
                                      longitude: Union[float, Decimal]) -> tuple[str, dict[str, Decimal]]:
        # Calculate mesh_code & base_point
        mesh_code = f"{int(Decimal(latitude) * Decimal(1.5)):02}{int(Decimal(longitude) % Decimal(100)):02}"
        base_point = dict(latitude=Decimal(float(mesh_code[:2]) / 1.5), longitude=Decimal(float(f"1{mesh_code[2:]}")))

        for i in range(1, self.order):
            mesh_indice = [(Decimal(latitude) - base_point['latitude']) // self.size_array[i]['latitude'],
                           (Decimal(longitude) - base_point['longitude']) // self.size_array[i]['longitude']]
            mesh_code += f"{mesh_indice[0]}{mesh_indice[1]}"

            # Update base_point
            base_point['latitude'] += self.size_array[i]['latitude'] * Decimal(mesh_indice[0])
            base_point['longitude'] += self.size_array[i]['longitude'] * Decimal(mesh_indice[1])
        return mesh_code, base_point

    def calculate_mesh_point_from_code(self, mesh_code: int) -> tuple[Decimal]:
        x = y = counter = 0
        code = str(mesh_code)
        for order in range(1, self.order + 1):
            x_code = Decimal(code[2:4]) if order == 1 else Decimal(code[counter + 1])
            y_code = Decimal(code[:2]) if order == 1 else Decimal(code[counter])
            counter += 4 if order == 1 else 2
            x += x_code + Decimal(100) if order == 1 else x_code * self.size_array[order - 1]['longitude']
            y += y_code / Decimal(1.5) if order == 1 else y_code * self.size_array[order - 1]['latitude']
        return x, y


# 分割地域メッシュ
class DivideRegionMesh(RegionMesh):

    def __post_init__(self) -> None:
        self.size_array = self.create_mesh_size_array(self.order)
        self.base_mesh = StandardRegionMesh(3)

    @staticmethod
    def create_mesh_size_array(order: int) -> list[dict[str, Decimal]]:
        size_array = [dict(latitude=Decimal(15. / 3600.), longitude=Decimal(22.5 / 3600.))]
        for _ in range(1, order):
            coef = 2.
            size_array.append(
                dict(latitude=Decimal(size_array[-1]['latitude'] / Decimal(coef)),
                     longitude=Decimal(size_array[-1]['longitude'] / Decimal(coef))))
        return size_array

    def get_mesh_code(self, latitude: Union[float, Decimal], longitude: Union[float, Decimal]) -> int:
        mesh_code, base_point = self.base_mesh.calculate_mesh_code_and_point(latitude, longitude)  # 3rd mesh

        for i in range(self.order):
            mesh_indice = [(Decimal(latitude) - base_point['latitude']) // self.size_array[i]['latitude'],
                           (Decimal(longitude) - base_point['longitude']) // self.size_array[i]['longitude']]
            mesh_code += f"{mesh_indice[0] * 2 + mesh_indice[1] + 1}"

            # Update base_point
            base_point['latitude'] += self.size_array[i]['latitude'] * Decimal(mesh_indice[0])
            base_point['longitude'] += self.size_array[i]['longitude'] * Decimal(mesh_indice[1])
        return int(mesh_code)

    def calculate_mesh_point_from_code(self, mesh_code: int) -> Polygon:
        x, y = self.base_mesh.calculate_mesh_point_from_code(int(str(mesh_code)[:8]))
        counter = 0
        code = str(mesh_code)[8:]
        for order in range(self.order):
            x_code = Decimal((int(code[counter]) - 1) % 2)
            y_code = Decimal((int(code[counter]) - 1) // 2)
            counter += 1
            x += x_code * self.size_array[order]['longitude']
            y += y_code * self.size_array[order]['latitude']
        return x, y


# 統合地域メッシュ
class MergedRegionMesh(RegionMesh):

    def __post_init__(self) -> None:
        self.size_array = self.create_mesh_size_array(self.order)
        self.base_mesh = StandardRegionMesh(2)

    @staticmethod
    def create_mesh_size_array(order: int) -> list[dict[str, Decimal]]:
        size_array = [dict(latitude=Decimal(1. / 24.), longitude=Decimal(1. / 16.))]  # 5x regional mesh
        if order > 1:
            size_array.append(dict(latitude=Decimal(1. / 60.), longitude=Decimal(1. / 40.)))  # 2x regional mesh
        return size_array

    def get_mesh_code(self, latitude: Union[float, Decimal], longitude: Union[float, Decimal]) -> int:
        mesh_code, base_point = self.base_mesh.calculate_mesh_code_and_point(latitude, longitude)  # 2nd mesh
        mesh_indice = [(Decimal(latitude) - base_point['latitude']) // self.size_array[self.order - 1]['latitude'],
                       (Decimal(longitude) - base_point['longitude']) // self.size_array[self.order - 1]['longitude']]

        if self.order == 1:
            mesh_code += f"{mesh_indice[0] * 2 + mesh_indice[1] + 1}"
        else:
            mesh_code += f"{mesh_indice[0] * 2}{mesh_indice[1] * 2}5"
        return int(mesh_code)

    def calculate_mesh_point_from_code(self, mesh_code: int) -> Polygon:
        x, y = self.base_mesh.calculate_mesh_point_from_code(int(str(mesh_code)[:6]))
        code = str(mesh_code)[6:]

        if self.order == 1:
            x_code = Decimal((int(code) - 1) % 2)
            y_code = Decimal((int(code) - 1) // 2)
        else:
            x_code = Decimal(int(code[1]) // 2)
            y_code = Decimal(int(code[0]) // 2)
        x += x_code * self.size_array[self.order - 1]['longitude']
        y += y_code * self.size_array[self.order - 1]['latitude']
        return x, y


# 地域メッシュ作成クラス
class RegionMeshCreator:

    @classmethod
    def create(cls, order: int) -> RegionMesh:
        if order < 4:
            mesh = StandardRegionMesh(order)
        elif order < 7:
            mesh = DivideRegionMesh(order - 3)
        else:
            mesh = MergedRegionMesh(order - 6)
        return mesh


def get_intersect_mesh(aoi: Polygon, order: int) -> gpd.GeoDataFrame:
    return RegionMeshCreator.create(order).get_intersect_mesh(aoi)
