import { StyleSheet, Text, View } from "react-native";
import React, { useEffect } from "react";
import { useNavigation, useLocalSearchParams, useRouter } from "expo-router";

export default function Chat() {
    const navigation = useNavigation();
    const { param, name } = useLocalSearchParams();
    useEffect(() => {
        navigation.setOptions({
            headerLeft: () =>
            (
                <View>
                    <Text>{name}{name}</Text>
                </View>
            )
        })
    }, [])
    return (
        <View>
            <Text>{param}</Text>
        </View>
    );
}

const styles = StyleSheet.create({});
